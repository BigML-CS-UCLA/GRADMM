# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task."""

import dataclasses
from dataclasses import asdict
import importlib.metadata
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import time
from typing import Callable, Dict, List, Optional, TYPE_CHECKING, Tuple, Union
# isort: on
import numpy as np
from packaging import version
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import (
    Dataset,
    RandomSampler,
)
from tqdm import tqdm
from transformers.data.data_collator import DataCollator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import hp_params
from transformers.integrations.deepspeed import (
    deepspeed_init,
    deepspeed_load_checkpoint,
    is_deepspeed_available,
)
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    get_dataloader_sampler,
    get_model_param_count,
)
from transformers.trainer_utils import (
    EvalPrediction,
    HPSearchBackend,
    PREFIX_CHECKPOINT_DIR,
    TrainOutput,
    has_length,
    speed_metrics,
)
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import (
    XLA_FSDPV2_MIN_VERSION,
    is_accelerate_available,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
    logging,
)

from utils import end_accumulate_time, start_accumulate_time


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(
        XLA_FSDPV2_MIN_VERSION
    )
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse(
        "1.10"
    )

    from .trainer_pt_utils import (
        smp_forward_backward,
        smp_forward_only,
        smp_gather,
        smp_nested_concat,
    )
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedType,
        save_fsdp_model,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse(
            "0.7.0"
        ):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


def _get_fsdp_ckpt_kwargs():
    # TODO: @AjayP13, @younesbelkada replace this check with version check at the next `accelerate` release
    if is_accelerate_available() and "adapter_only" in list(
        inspect.signature(save_fsdp_model).parameters
    ):
        return {"adapter_only": True}
    else:
        return {}


if TYPE_CHECKING:
    import optuna

    if is_datasets_available():
        import datasets

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

from transformers import Trainer


class OurTrainer(Trainer):
    """Attributes:

    test_dataset:
    training_framework:
    eval_samples:
    test_samples:
    train_samples:
    control:
    model_wrapped:
    lr_scheduler:
    optimizer:
    state:
    model:
    deepspeed:
    current_flos:
    is_in_train:
    named_parameters_to_optim:
    zo_random_seed:
    projected_grad:
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        test_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        training_framework: Optional[Dataset] = None,
        eval_samples: Optional[list] = None,
        test_samples: Optional[list] = None,
        train_samples: Optional[list] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[
            torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR
        ] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("type of optimizer: ", type(optimizers[0]))
        self.test_dataset = test_dataset
        self.training_framework = training_framework
        self.eval_samples = eval_samples
        self.test_samples = test_samples
        self.train_samples = train_samples
        args_dict = asdict(self.args)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(args_dict, f, indent=2)
        self.args.main_results_summary = args_dict
        self.args.main_results = {
            "args": args_dict,
        }

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)
        if not self.args.no_save_weights:
            self.save_model(output_dir, _internal_call=True)

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            # self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
            with open(
                os.path.join(output_dir, TRAINER_STATE_NAME), "w", encoding="utf-8"
            ) as f:
                f.write(
                    json.dumps(dataclasses.asdict(self.state), indent=2, sort_keys=True)
                    + "\n"
                )
            with open(os.path.join(output_dir, "main_results.json"), "w") as f:
                json.dump(self.args.main_results, f, indent=2)

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval
    ):
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.detach().item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            eval_metrics = self.our_evaluate(
                eval_dataset=self.eval_dataset,
                eval_samples=self.eval_samples,
                ignore_keys=ignore_keys_for_eval,
                metric_key_prefix="eval",
            )
            test_metrics = self.our_evaluate(
                eval_dataset=self.test_dataset,
                eval_samples=self.test_samples,
                ignore_keys=ignore_keys_for_eval,
                metric_key_prefix="test",
            )
            if self.args.report_train:
                train_metrics = self.our_evaluate(
                    eval_dataset=self.train_dataset,
                    eval_samples=self.train_samples,
                    ignore_keys=ignore_keys_for_eval,
                    metric_key_prefix="total_train",
                )
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            ##################UPDATE MAIN RESULTS######################
            eval_metric_hist = []
            test_metric_hist = []
            total_train_metric_hist = []
            for item in self.state.log_history:
                if "eval_loss" in item.keys():
                    if "eval_accuracy" in item.keys():
                        eval_metric_hist.append((
                            item["eval_accuracy"],
                            item["step"],
                            item["eval_per_class_accuracy"],
                        ))
                    elif "eval_f1" in item.keys():
                        eval_metric_hist.append((item["eval_f1"], item["step"]))
                elif "test_loss" in item.keys():
                    if "test_accuracy" in item.keys():
                        test_metric_hist.append((
                            item["test_accuracy"],
                            item["step"],
                            item["test_per_class_accuracy"],
                        ))
                    elif "test_f1" in item.keys():
                        test_metric_hist.append((item["test_f1"], item["step"]))
                elif "total_train_loss" in item.keys():
                    if "total_train_accuracy" in item.keys():
                        total_train_metric_hist.append(
                            (item["total_train_accuracy"], item["step"])
                        )
                    elif "total_train_f1" in item.keys():
                        total_train_metric_hist.append(
                            (item["total_train_f1"], item["step"])
                        )

            eval_metric_hist.sort(key=lambda x: x[0], reverse=True)
            test_metric_hist.sort(key=lambda x: x[0], reverse=True)
            total_train_metric_hist.sort(key=lambda x: x[0], reverse=True)

            self.args.main_results["best_valid_acc"] = (
                eval_metric_hist[0][0] if len(eval_metric_hist) > 0 else None
            )
            self.args.main_results["best_valid_step"] = (
                eval_metric_hist[0][1] if len(eval_metric_hist) > 0 else None
            )
            self.args.main_results["best_valid_per_class_acc"] = (
                eval_metric_hist[0][2] if len(eval_metric_hist) > 0 else None
            )
            self.args.main_results["best_test_metric"] = (
                test_metric_hist[0][0] if len(test_metric_hist) > 0 else None
            )
            self.args.main_results["best_test_step"] = (
                test_metric_hist[0][1] if len(test_metric_hist) > 0 else None
            )
            self.args.main_results["best_test_per_class_acc"] = (
                test_metric_hist[0][2] if len(test_metric_hist) > 0 else None
            )
            self.args.main_results["best_total_train_acc"] = (
                total_train_metric_hist[0][0]
                if len(total_train_metric_hist) > 0
                else None
            )
            self.args.main_results["best_total_train_step"] = (
                total_train_metric_hist[0][1]
                if len(total_train_metric_hist) > 0
                else None
            )

            if len(test_metric_hist) > 0:
                best_dev_step = eval_metric_hist[0][1]
                for test_metric, step, _ in test_metric_hist:
                    if step == best_dev_step:
                        self.args.main_results["test_metric_given_best_dev_metric"] = (
                            test_metric
                        )
                        break

            eval_metric_hist.sort(key=lambda x: x[1], reverse=False)
            test_metric_hist.sort(key=lambda x: x[1], reverse=False)
            total_train_metric_hist.sort(key=lambda x: x[1], reverse=False)

            self.args.main_results["valid_acc_hist"] = eval_metric_hist
            self.args.main_results["test_acc_hist"] = test_metric_hist
            self.args.main_results["total_train_acc_hist"] = total_train_metric_hist

            # get all train loss
            train_loss_hist = []
            eval_loss_hist = []
            test_loss_hist = []
            total_train_loss_hist = []
            for item in self.state.log_history:
                if "loss" in item.keys():
                    train_loss_hist.append((item["loss"], item["step"]))
                if "eval_loss" in item.keys():
                    eval_loss_hist.append((item["eval_loss"], item["step"]))
                if "test_loss" in item.keys():
                    test_loss_hist.append((item["test_loss"], item["step"]))
                if "total_train_loss" in item.keys():
                    total_train_loss_hist.append((item["total_train_loss"], item["step"]))
            self.args.main_results["train_loss_hist"] = train_loss_hist
            self.args.main_results["valid_loss_hist"] = eval_loss_hist
            self.args.main_results["test_loss_hist"] = test_loss_hist
            self.args.main_results["total_train_loss_hist"] = total_train_loss_hist
            ##################UPDATE MAIN RESULTS END######################

            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def our_evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        eval_samples: Optional[list] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute
        metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*): Pass a dataset if you wish to
            override `self.eval_dataset`. If it is a [`~datasets.Dataset`],
            columns not accepted by the `model.forward()` method are automatically
            removed. It must implement the `__len__` method.
            ignore_keys (`List[str]`, *optional*): A list of keys in the output of
            your model (if it is a dictionary) that should be ignored when
            gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`): An optional
            prefix to be used as the metrics key prefix. For example the metrics
            "bleu" will be named "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics
            computed from the predictions. The
            dictionary also contains the epoch number which comes from the training
            state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics,
            # otherwise we defer to self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        acc_metrics = self.training_framework.evaluate([], eval_samples)
        for key, val in acc_metrics.items():
            output.metrics[f"{metric_key_prefix}_{key}"] = val

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA
            # (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    from transformers.trainer_pt_utils import (
        _get_learning_rate,
        log_metrics,
        metrics_format,
        save_metrics,
        save_state,
    )

    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # MeZO added: Linear probing
        if self.args.linear_probing:

            def _get_token_prediction_layer(model):
                if model.config.model_type == "opt":
                    return model.lm_head
                else:
                    raise NotImplementedError(model.config.model_type)

            def _extract_features(model, *args, **kwargs):
                """some magic for getting features pre last layer"""
                features = {}

                def __hook(model_, input_, output_):
                    features["features"] = input_[0].detach()

                _get_token_prediction_layer(model).register_forward_hook(__hook)
                model.forward(*args, **kwargs)
                return features["features"]

            logger.info("Linear probing")
            logger.info("Starting to get features for training dataset")
            targets = []
            features = []
            with torch.inference_mode():
                for step, inputs in enumerate(tqdm(train_dataloader)):
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)

                feature = _extract_features(self.model, **inputs)
                target = inputs["labels"]

                # Shift the target (bc it's autoregressive LM) and add the corresponding part
                assert (
                    not self.args.train_as_classification
                    and self.args.only_train_option
                )
                feature, target = feature[:, :-1], target[:, 1:]
                for _i, _len in enumerate(inputs["option_len"]):
                    features.append(feature[_i, -_len:])
                    targets.append(target[_i, -_len:])

            logger.info("Finished getting features for training dataset")

            features = torch.cat(features, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()
            # Whether to use bias
            if self.model.config.model_type in ["opt", "gpt2"]:
                use_bias = False
            else:
                raise NotImplementedError
            # Set early stopping
            tol = (
                0.01 if self.args.lp_early_stopping else 1e-4
            )  # 1e-4 is scipy default
            max_iter = 1000 if self.args.lp_early_stopping else 5000

            logger.info("Fitting logistic regression...")
            reg = LogisticRegressionCV(
                max_iter=max_iter,
                fit_intercept=use_bias,
                multi_class="multinomial",
                random_state=0,
                tol=tol,
                n_jobs=-1,
            ).fit(features, targets)
            logger.info("Done")

            logger.info("Assigning weights to model")
            decoder = _get_token_prediction_layer(self.model)
            coef_torch = torch.tensor(
                reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype
            )
            if use_bias:
                bias_torch = torch.tensor(
                    reg.intercept_,
                    device=decoder.weight.device,
                    dtype=decoder.weight.dtype,
                )
            if coef_torch.shape[0] == 1:  # The regressor only detects two classes
                assert len(reg.classes_) == 2
                coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
                if use_bias:
                    bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)

            for _i, token_id in enumerate(reg.classes_):
                decoder.weight.data[token_id] = coef_torch[_i]
                if use_bias:
                    decoder.bias.data[token_id] = bias_torch[_i]

            return None

        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(
                        1, self.args.n_gpu
                    )
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(
            f"Currently training with a batch size of: {self._train_batch_size}"
        )
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = (
            self._train_batch_size
            * args.gradient_accumulation_steps
            * args.world_size
        )

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = (
                len_dataloader // args.gradient_accumulation_steps
            )
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps)
                        * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(
                    args.num_train_epochs * num_update_steps_per_epoch
                )
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = (
                    self.num_examples(train_dataloader) * args.num_train_epochs
                )
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader) * args.num_train_epochs
                    )
        elif (
            args.max_steps > 0
        ):  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = (
                    self.num_tokens(train_dataloader, args.max_steps)
                    * args.gradient_accumulation_steps
                )
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does"
                f" not have a length, was {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP."
                    " Please use DDP (torchrun or torch.distributed.launch"
                    " (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            is_sagemaker_mp_enabled()
            or self.is_fsdp_xla_enabled
            or self.is_fsdp_enabled
        )

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps
            )

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(
                        self.model, self.optimizer
                    )
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped,
                    resume_from_checkpoint,
                    load_module_strict=not _is_peft_model(self.model),
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(
            "  Instantaneous batch size per device ="
            f" {self.args.per_device_train_batch_size:,}"
        )
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(
                "  Training with DataParallel so batch size has been adjusted to:"
                f" {self._train_batch_size:,}"
            )
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) ="
            f" {total_train_batch_size:,}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(
            "  Number of trainable parameters ="
            f" {get_model_param_count(model, trainable_only=True):,}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved"
                " global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = (
                trial.assignments
                if self.hp_search_backend == HPSearchBackend.SIGOPT
                else trial
            )
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None

        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            if (
                epoch == epochs_trained
                and resume_from_checkpoint is not None
                and steps_trained_in_current_epoch == 0
            ):
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(
                    epoch_iterator, steps_trained_in_current_epoch
                )
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                start_accumulate_time()
                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current"
                            " model is not configured properly to know what item is the"
                            " input. To fix this, add a `main_input_name` attribute to the"
                            " model class you are using."
                        )
                    else:
                        input_device = inputs[main_input_name].device
                        self.state.num_input_tokens_seen += torch.sum(
                            self.accelerator.gather(
                                torch.tensor(
                                    inputs[main_input_name].numel(),
                                    device=input_device,
                                    dtype=torch.int64,
                                )
                            )
                        ).item()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )

                # MeZO added: estimate gradient
                with self.accelerator.accumulate(model):
                    if args.trainer == "sgd_in_place":
                        tr_loss_step = self.training_step(model, inputs)
                    else:
                        tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (
                        1 + self.state.global_step - self._globalstep_last_logged
                    )
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            "Calculated loss must be on the original device:"
                            f" {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)
                    if args.trainer == "sgd_in_place":
                        model.zero_grad()
                    else:

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(
                                    args.max_grad_norm
                                )
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type
                                == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        # Optimizer step
                        self.optimizer.step()
                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(
                                self.lr_scheduler,
                                torch.optim.lr_scheduler.ReduceLROnPlateau,
                            ):
                                self.lr_scheduler.step()
                        model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control
                    )
                    end_accumulate_time()

                    self._maybe_log_save_evaluate(
                        tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control
                    )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator,"
                    f" stopping training at step {self.state.global_step}! This is"
                    " expected if you're using an IterableDataset and set num_steps"
                    f" ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control
            )
            self._maybe_log_save_evaluate(
                tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval
            )

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU"
                        " configured. Check your training configuration if this is"
                        " unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on"
            " huggingface.co/models =)\n\n"
        )
        if (
            args.load_best_model_at_end
            and self.state.best_model_checkpoint is not None
        ):
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(
            self.state.global_step, 0.001
        )  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir
        )

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if (
            self.args.should_save
            and self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
        ):
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to"
                        " args.save_total_limit"
                    )
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)
        # self.state.save_to_json(
        #    os.path.join(self.args.output_dir, TRAINER_STATE_NAME)
        # )
        with open(
            os.path.join(self.args.output_dir, TRAINER_STATE_NAME),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(dataclasses.asdict(self.state), indent=2, sort_keys=True)
                + "\n"
            )

        ################## UPDATE MAIN RESULTS ######################
        eval_metric_hist = []
        test_metric_hist = []
        total_train_metric_hist = []
        for item in self.state.log_history:
            if "eval_loss" in item.keys():
                if "eval_accuracy" in item.keys():
                    eval_metric_hist.append((
                        item["eval_accuracy"],
                        item["step"],
                        item["eval_per_class_accuracy"],
                    ))
                elif "eval_f1" in item.keys():
                    eval_metric_hist.append((item["eval_f1"], item["step"]))
            elif "test_loss" in item.keys():
                if "test_accuracy" in item.keys():
                    test_metric_hist.append((
                        item["test_accuracy"],
                        item["step"],
                        item["test_per_class_accuracy"],
                    ))
                elif "test_f1" in item.keys():
                    test_metric_hist.append((item["test_f1"], item["step"]))
            elif "total_train_loss" in item.keys():
                if "total_train_accuracy" in item.keys():
                    total_train_metric_hist.append(
                        (item["total_train_accuracy"], item["step"])
                    )
                elif "total_train_f1" in item.keys():
                    total_train_metric_hist.append((item["total_train_f1"], item["step"]))

        eval_metric_hist.sort(key=lambda x: x[0], reverse=True)
        test_metric_hist.sort(key=lambda x: x[0], reverse=True)
        total_train_metric_hist.sort(key=lambda x: x[0], reverse=True)

        self.args.main_results["best_valid_acc"] = (
            eval_metric_hist[0][0] if len(eval_metric_hist) > 0 else None
        )
        self.args.main_results["best_valid_step"] = (
            eval_metric_hist[0][1] if len(eval_metric_hist) > 0 else None
        )
        self.args.main_results["best_valid_per_class_acc"] = (
            eval_metric_hist[0][2] if len(eval_metric_hist) > 0 else None
        )
        self.args.main_results["best_test_metric"] = (
            test_metric_hist[0][0] if len(test_metric_hist) > 0 else None
        )
        self.args.main_results["best_test_step"] = (
            test_metric_hist[0][1] if len(test_metric_hist) > 0 else None
        )
        self.args.main_results["best_test_per_class_acc"] = (
            test_metric_hist[0][2] if len(test_metric_hist) > 0 else None
        )
        self.args.main_results["best_total_train_acc"] = (
            total_train_metric_hist[0][0]
            if len(total_train_metric_hist) > 0
            else None
        )
        self.args.main_results["best_total_train_step"] = (
            total_train_metric_hist[0][1]
            if len(total_train_metric_hist) > 0
            else None
        )

        if len(test_metric_hist) > 0:
            best_dev_step = eval_metric_hist[0][1]
            for test_metric, step, _ in test_metric_hist:
                if step == best_dev_step:
                    self.args.main_results["test_metric_given_best_dev_metric"] = (
                        test_metric
                    )
                    break

        eval_metric_hist.sort(key=lambda x: x[1], reverse=False)
        test_metric_hist.sort(key=lambda x: x[1], reverse=False)
        total_train_metric_hist.sort(key=lambda x: x[1], reverse=False)

        self.args.main_results["valid_acc_hist"] = eval_metric_hist
        self.args.main_results["test_acc_hist"] = test_metric_hist
        self.args.main_results["total_train_acc_hist"] = total_train_metric_hist

        # get all train loss
        train_loss_hist = []
        eval_loss_hist = []
        test_loss_hist = []
        total_train_loss_hist = []
        for item in self.state.log_history:
            if "loss" in item.keys():
                train_loss_hist.append((item["loss"], item["step"]))
            if "eval_loss" in item.keys():
                eval_loss_hist.append((item["eval_loss"], item["step"]))
            if "test_loss" in item.keys():
                test_loss_hist.append((item["test_loss"], item["step"]))
            if "total_train_loss" in item.keys():
                total_train_loss_hist.append((item["total_train_loss"], item["step"]))
        self.args.main_results["train_loss_hist"] = train_loss_hist
        self.args.main_results["valid_loss_hist"] = eval_loss_hist
        self.args.main_results["test_loss_hist"] = test_loss_hist
        self.args.main_results["total_train_loss_hist"] = total_train_loss_hist
        ##################UPDATE MAIN RESULTS END######################

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _set_signature_columns_if_needed(self):
        """We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task"""
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(
                set(["label", "label_ids"] + self.label_names)
            )
            self._signature_columns += ["gold"]
