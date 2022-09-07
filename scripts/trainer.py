# -*- encoding: utf-8 -*-
'''
@File    :   Trainer.py
@Time    :   2022/04/18 09:03:32
@Author  :   Yuan Wind
@Desc    :   None
'''
import collections
from transformers.trainer import *
from modules.nn.adversarial import AWP, EMA, FGM, PGD
import logging

logger = logging.getLogger(__name__.replace('_', ''))

class MyTrainer(Trainer):
    def __init__(self,config, evaluater=None, **kwargs):
        super().__init__(args=config.trainer_args, **kwargs)
        self.config = config
        self.evaluater = evaluater
        self.current_step = 0
        # TODO early_stop 换成Trainer自带的更好
        self.early_stop_mode = self.config.early_stop_mode # -1代表关闭，0代表连续评测四次没提升就停止，>0的数字代表具体的哪一轮停止
        self.early_stop_counter = 0 
        self.use_swa = False
        self.use_lookahead = False
        
    def loss_backward(self,loss):
        """
        非原Trainer有的
        """
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
    
    
    def create_optimizer(self):
        """
        设置分层学习率、swa、lookahead
        """
        
            
        if self.optimizer is None:
            
            optimizer_grouped_parameters = self.model.optimizer_grouped_parameters(self.args.weight_decay)
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        if self.args.other_tricks is not None:
            if 'swa' in self.args.other_tricks:
                from torchcontrib.optim import SWA
                # https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
                logger.info(f'---------------- Using SWA, swa_start={self.args.swa_start}, swa_freq={self.args.swa_freq}, swa_lr={self.args.swa_lr}-------------')
                self.use_swa = True
                self.optimizer=SWA(self.optimizer,swa_start=self.args.swa_start,swa_freq=self.args.swa_freq,swa_lr=self.args.swa_lr)
            elif 'lookahead' in self.args.other_tricks:
                from modules.nn.lookahead import Lookahead
                # https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
                logger.info(f'---------------- Using Lookahead, lookahead_k={self.args.lookahead_k}, lookahead_alpha={self.args.lookahead_alpha}--------------------')
                self.use_lookahead = True
                self.optimizer = Lookahead(self.optimizer, k=self.args.lookahead_k, alpha=self.args.lookahead_alpha) # Initialize Lookahead


        return self.optimizer
    
    
    def attack_step(self, model, inputs):
        if self.args.adversarival_type == 'fgm':
                self.adversarival.attack(epsilon=self.args.fgm_e, emb_name=self.args.emb_name)
                with self.autocast_smart_context_manager():
                    loss_adv = self.compute_loss(model, inputs)
                self.loss_backward(loss_adv)
                self.adversarival.restore(emb_name=self.args.emb_name)
        elif self.args.adversarival_type == 'pgd':
            self.adversarival.backup_grad()
            for t in range(self.args.pgd_k):
                self.adversarival.attack(epsilon=self.args.pgd_e,
                                            alpha=self.args.pgd_a,
                                            emb_name=self.args.emb_name,
                                            is_first_attack=(t==0)
                                            )
                if t != self.args.pgd_k-1:
                    self.optimizer.zero_grad()
                else:
                    self.adversarival.restore_grad()
                with self.autocast_smart_context_manager():
                    loss_adv = self.compute_loss(model, inputs)
                self.loss_backward(loss_adv)
            self.adversarival.restore(emb_name=self.args.emb_name)
        
        elif self.args.adversarival_type == 'awp':
            if self.adversarival.adv_lr == 0:
                return None
            self.adversarival.save()
            for _ in range(self.adversarival.adv_step):
                self.adversarival.attack()
                with self.autocast_smart_context_manager():
                    loss_adv = self.compute_loss(model, inputs)
                self.optimizer.zero_grad()
                self.loss_backward(loss_adv)
            self.adversarival.restore()
        return loss_adv
            
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]):
        """
        重写了该方法，主要加入了 FGM、PGD、EMA
        """
        model.train()
        if self.state.global_step >= self.current_step: # 说明上一个step更新了参数，而不是梯度累积的step
            self.current_step = self.state.global_step
            if self.ema is not None and self.state.global_step == self.args.ema_start_steps:
                self.ema.register()
            if self.ema is not None and self.state.global_step > self.args.ema_start_steps:
                self.ema.update()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        self.loss_backward(loss)
        
        # 对抗训练
        if self.adversarival is not None and self.state.global_step >= self.args.adv_start_steps:
            loss = self.attack_step(model, inputs)
        
        # 最后一轮最后一步运行完之后，如果使用了SWA，则要进行opt.swap_swa_sgd()和 opt.bn_update(train_loader, model)
        if self.use_swa and (self.state.global_step+1) == self.state.max_steps:
            logger.info('Running opt.swap_swa_sgd()和 opt.bn_update(train_loader, model)')
            self.optimizer.swap_swa_sgd()
            self.optimizer.bn_update(self.get_train_dataloader(),model)
            
        return loss.detach()


    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ):
        """
        Trainer自带的该方法是将验证集或者测试集全部预测完之后，将全部的logits、labels收集起来。
        之后再传到Evaluater里计算metric。可能会浪费大量的显存或者内存。
        此处MyTrainer增加了每一个step的后处理操作。将一定的steps预测的结果写入文件而不是全部的,然后在evaluate里读取文件计算metric
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            
        if self.ema is not None:
            self.ema.apply_shadow()
            
        # 保存ema之后的参数
        
        
        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        
        batch_size = self.args.per_device_eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")
        logger.info(f"  Current epoch = {self.state.epoch}")
        
        
        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode =  None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                self.evaluater.steps_evaluate(logits, inputs_decode, labels)
                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
        if preds_host is not None:
            self.evaluater.steps_evaluate(preds_host=logits, inputs_host=inputs_decode, labels_host=labels)        
        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        # if all_preds is not None:
        #     all_preds = nested_truncate(all_preds, num_samples)
        # if all_labels is not None:
        #     all_labels = nested_truncate(all_labels, num_samples)
        # if all_inputs is not None:
        #     all_inputs = nested_truncate(all_inputs, num_samples)

        # Metrics!
        if self.compute_metrics is not None:
            metrics = self.compute_metrics()
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
                
        # 只保存评测分数最高的模型或参数，而不保存optimizer等信息
        best_key = f'{metric_key_prefix}_{args.metric_for_best_model}' 
        if self.state.best_metric is not None: # 第一轮未评测之前，其为 None
            if best_key in metrics: 
                if (metrics[best_key] > self.state.best_metric and self.args.greater_is_better) or \
                   (metrics[best_key] < self.state.best_metric and not self.args.greater_is_better): 
                    logger.info(f'{best_key} improved from {self.state.best_metric} to {metrics[best_key]}.')
                    self.save()
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
            else:
                logger.warning(f'{best_key} not in metrics.keys(): {metrics}')
        else:
            self.save()
        if self.ema is not None:
            self.ema.restore()
        
        if self.early_stop_mode > 0 and self.state.epoch >= self.early_stop_mode:
            logger.info(f'Current epoch: {self.state.epoch}, stop training because of early_stop_mode = {self.early_stop_mode}.')
            self.control.should_training_stop = True
        elif self.early_stop_mode == 0 and self.early_stop_counter == 4:
            logger.info(f'Current epoch: {self.state.epoch}, stop training because of early_stop_mode = {self.early_stop_mode} and early_stop_counter = {self.early_stop_counter}.')
            self.control.should_training_stop = True
            
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def save(self, only_params = True):
        # 增加了该方法供自定义保存使用。默认save方式为保存最好的self.state.best_metric的模型参数到指定的 save_path
        save_path = self.config.best_model_file
        
        if only_params:
            params_state = self.model.state_dict()
            new_params_state = params_state.copy()
            for k, v in params_state.items():
                if 'positions_encoding' in k: # nezha的positions_encoding不用保存
                    new_params_state.pop(k)
                    logger.warning('Positions_encoding params will not be saved.')
            torch.save(new_params_state, save_path)
            logger.info(f'Save model params to {save_path}')
        
        else:
            torch.save(self.model, save_path)
            logger.info(f'Save total model to {save_path}')
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        # 重写该方法，主要是为了能够记录 Model 中自定义的一些状态，比如多个loss值之类的。
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            #----------------增加的代码：---------------------
            if self.model.cur_batch_state is not None:
                if 'loss' in self.model.cur_batch_state:
                    self.model.cur_batch_state['loss_in_model'] = self.model.cur_batch_state.pop('loss')
                logs.update(self.model.cur_batch_state)
            #------------------结束--------------------------
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
    
    def call_model_init(self, trial=None):
        # 增加对抗训练初始化
        model_init_argcount = number_of_arguments(self.model_init)
        if model_init_argcount == 0:
            model = self.model_init()
        elif model_init_argcount == 1:
            model = self.model_init(trial)
        else:
            raise RuntimeError("model_init should have 0 or 1 argument.")

        if model is None:
            raise RuntimeError("model_init should not return None.")

        # Tricks !
        
        self.adversarival = None
        self.ema = None

        if self.args.do_train:
            if self.args.adversarival_type == 'fgm':
                self.adversarival = FGM(model) 
                logger.info(f'------------Use FGM, fgm_e = {self.args.fgm_e}, emb_name = {self.args.emb_name}.---------------')
            elif self.args.adversarival_type == 'pgd':
                self.adversarival = PGD(model)
                logger.info(f'------------Use PGD, pgd_e: {self.args.pgd_e}, pdg_a:{self.args.pgd_a} , emb_name = {self.args.emb_name}.---------------')
            
            elif self.args.adversarival_type == 'awp':
                self.adversarival = AWP(model, self.args.awp_param, self.args.awp_a, self.args.apw_e, self.args.apw_k)
                logger.info(f'------------Use AWP, apw_e: {self.args.apw_e}, awp_a:{self.args.awp_a} , awp_param = {self.args.awp_param}.---------------')
            
            if self.args.other_tricks is not None and 'ema' in self.args.other_tricks:
                self.ema = EMA(model, self.args.ema_decay)
                logger.info(f'------------Use EMA, ema_decay: {self.args.ema_decay}, ema_start_steps:{self.args.ema_start_steps}.---------------')

        
        return model

        # def get_train_dataloader(self):
        #     pass
        # get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
            # pass
            
        # def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
            # pass
    
    # 以下代码未测试，仅供参考
    def ensemble_loops(self, models, test_set, prediction_loss_only=False):
        args = self.args
        metric_key_prefix = 'ens'
        dataloader = self.get_eval_dataloader(test_set)
        batch_size = self.args.per_device_eval_batch_size
        logger.info("***** Running Ensemble *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")
        logger.info(f"  Model num = {len(models)}")
        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # 模型集成
            logits = []
            loss = None
            for model in models:
            # Prediction step
                one_loss, logit, labels = self.prediction_step(model, inputs, prediction_loss_only)
                logits.append(logit[0] if type(logit) is tuple else logit)
                if one_loss is not None:
                    if loss is None:
                        loss = []
                    loss.append(one_loss)
            # 对模型的结果求平均
            logits = torch.stack(logits)
            logits = torch.mean(logits, dim=0)
            if loss is not None:
                loss = torch.stack(loss)
                loss = torch.mean(loss, dim = 0)
                
            inputs_decode =  None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                self.evaluater.steps_evaluate(logits, inputs_decode, labels)
                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
        if labels_host is not None:
            labels = nested_numpify(labels_host)

        self.evaluater.steps_evaluate(preds_host=logits, inputs_host=inputs_decode, labels_host=labels)        
        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]

        # Metrics!
        if self.compute_metrics is not None:
            metrics = self.compute_metrics()
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
                
    def model_soups(self, model_params_files, test_set, best_avg_params_file = 'best_avg_params.pt', is_ordered = False):
        """model soups：对不同超参下的模型参数进行参数平均 https://arxiv.org/abs/2203.05482

        Args:
            model_params_files (List[str]): 需要平均的模型参数文件, 需要按照各自验证集上的得分降序排列
            test_set (dataset): 验证集
            best_avg_params_file (str, optional): 最好的模型参数保存路径. Defaults to 'best_avg_params.pt'.
            is_ordered (bool): 是否按照测试集得分降序排列了，TODO 如果没有，则会进行评测并排序 
        """
        # orderd_params_files = []
        # for params_file in model_params_files[1:]:
        #     pass
        best_file = model_params_files[0]
        best_params = torch.load(best_file)
        self.model.load_state_dict(best_params, strict=False)
        best_metrics = self.evaluate(test_set)
        best_score = best_metrics['eval_F1']
        best_avg_params_list = [best_params]
        best_files = [best_file]
        best_avg_params = None
        avg_params = best_params.copy()
        final_res = []
        ignore_keys = ['LayerNorm', 'bias'] # 这两个参数不加入平均
        loop_cnt = 0
        total_loop_cnt = 0
        for params_file in model_params_files[1:]:
            for n in best_params.keys(): # 遍历每个模型的参数，觉得其是否加入平均
                if any(ignore_key in n for ignore_key in ignore_keys):
                    continue
                total_loop_cnt += 1
        for params_file in model_params_files[1:]:
            cur_params = torch.load(params_file)
            need_to_avg_params = best_avg_params_list + [cur_params]
            for n in best_params.keys(): # 遍历每个模型的参数，觉得其是否加入平均
                if any(ignore_key in n for ignore_key in ignore_keys):
                    continue
                loop_cnt += 1
                logger.info(f'Test {params_file}.{n} is need to be add or not...')
                cur_data = []
                for params in need_to_avg_params:
                    cur_data.append(params[n].data)
                params_bak = avg_params[n]
                avg_params[n] = torch.mean(torch.stack(cur_data), dim=0)
                self.model.load_state_dict(avg_params, strict=False)
                metrics = self.evaluate(test_set)
                cur_score = metrics['eval_F1']

                if cur_score > best_score:
                    best_score = cur_score
                    best_metrics = metrics
                    best_avg_params = avg_params
                    best_avg_params_list.append(cur_params)
                    best_files.append(params_file)
                    logger.info(f'Add {params_file}.{n} to averaged is better. {cur_score}')
                    final_res.append(f'{params_file}.{n}')
                else:
                    avg_params[n] = params_bak
                    logger.info(f'Add {params_file}.{n} to averaged is not better, so not. best:{best_score}, current:{cur_score}')
                logger.info(f'进度： {loop_cnt}/{total_loop_cnt}')
        
        logger.info(f'Best files to avg:{best_files}, best metric: {best_metrics}')
        torch.save(best_avg_params, best_avg_params_file)
        logger.info(f'Save the average params to {best_avg_params_file}')
        
    def ensemble_predict(self, model_files, test_set, evaluate_each_model = False):
        """Ensemble model_params_files 中的所有模型参数结果，按照batch进行集成
        Args:
            model_params_files (List[str]): 存储了模型参数的文件列表
            test_set (dataset): 预测的测试集
            
        """
        model_list = []
        for model_idx, model_file in enumerate(model_files):
            model_or_params = torch.load(model_file)
            if type(model_or_params) == collections.OrderedDict:
                model = self.call_model_init()
                model.load_state_dict(model_or_params, strict=False)
                model = model.to(device=self.args.device)
            else:
                model = model_or_params
            model.eval()
            if evaluate_each_model:
                metrics = self.evaluate(test_set)
                logger.info(f'{model_file} metrics:{metrics}.')
            model_list.append(model)
            logger.info(f'Load model_{model_idx} from {model_file}')
            
        logger.info(f'total models: {len(model_list)}')
        self.ensemble_loops(model_list, test_set)

            
        
        
        
        
        