from typing import Any, Optional, Tuple, Union, Dict
from warnings import warn
import cv2
import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch as th
import torch.distributed as dist
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from einops import rearrange, repeat
from einops import reduce
from data.genx_utils.labels import ObjectLabels
from data.utils.types import DataType, LstmStates, ObjDetOutput, DatasetSamplingMode, MambaStates
from models.detection.yolox.utils.boxes import postprocess
from models.detection.yolox_extension.models.detector import YoloXDetector
from utils.evaluation.prophesee.evaluator import PropheseeEvaluator
from utils.evaluation.prophesee.io.box_loading import to_prophesee
from utils.padding import InputPadderFromShape
from .utils.detection import BackboneFeatureSelector, EventReprSelector, RNNStates, Mode, mode_2_string, \
    merge_mixed_batches

BBOX_DTYPE = np.dtype({'names': ['t', 'x', 'y', 'w', 'h', 'class_id', 'track_id', 'class_confidence'],
                       'formats': ['<i8', '<f4', '<f4', '<f4', '<f4', '<u4', '<u4', '<f4'],
                       'offsets': [0, 8, 12, 16, 20, 24, 28, 32], 'itemsize': 40})

def draw_bboxes(img, boxes) -> None:
    """
    draw bboxes in the image img
    """
    # colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    # colors = [tuple(*item) for item in colors.tolist()]

    for i in range(boxes.shape[0]):
        pt1 = (int(boxes['x'][i]), int(boxes['y'][i]))
        size = (int(boxes['w'][i]), int(boxes['h'][i]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        score = boxes['class_confidence'][i]
        class_id = boxes['class_id'][i]
        if class_id == 0:
            class_name = "pedestrian"
            color = (255,0,0)
        elif class_id == 1:
            class_name = "car"
            color = (0,255,0)
        elif class_id == 2:
            class_name = "two wheeler"
            color = (255,153,255)
        # class_name = labelmap[class_id % len(labelmap)]
        # color = colors[class_id * 60 % 255]
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.rectangle(img, pt1, pt2, color, 2)
        # cv2.putText(img, class_name, (center[0], pt2[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        # cv2.putText(img, str(score), (center[0], pt1[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        text = '{}:{:.1f}%'.format(class_name, score * 100)
        txt_color = (0, 0, 0) if np.mean(np.array(color)/255) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        txt_bk_color = (np.array(color)/255 * 255 * 0.7).astype(np.uint8).tolist()
        txt_bk_color = color
        cv2.rectangle(
            img,
            (pt1[0], pt1[1] - 1),
            (pt1[0] + txt_size[0] + 1, pt1[1] - int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (pt1[0], pt1[1] - int(0.5 * txt_size[1])), font, 0.4, txt_color, thickness=1)
    return img
def draw_bboxes2(img, boxes) -> None:
    """
    draw bboxes in the image img
    """
    # colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    # colors = [tuple(*item) for item in colors.tolist()]

    for i in range(boxes.shape[0]):
        pt1 = (int(boxes['x'][i]), int(boxes['y'][i]))
        size = (int(boxes['w'][i]), int(boxes['h'][i]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        score = boxes['class_confidence'][i]
        class_id = boxes['class_id'][i]
        # 1MPX
        if class_id == 0:
            class_name = "pedestrian"
            color = (255,0,0)#(102,255,102)
        if class_id == 1:
            class_name = "two wheeler"
            color = (255,153,255)
        if class_id == 2:
            class_name = "car"
            color = (0,255,0)
        # if class_id == 0:
        #     class_name = "pedestrian"
        #     color = (255,0,0)
        # elif class_id == 1:
        #     class_name = "car"
        #     color = (0,255,0)
        # elif class_id == 2:
        #     class_name = "two wheeler"
        #     color = (255,153,255)
        # class_name = labelmap[class_id % len(labelmap)]
        # color = colors[class_id * 60 % 255]
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.rectangle(img, pt1, pt2, color, 2)
        # # cv2.putText(img, class_name, (center[0], pt2[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        # # cv2.putText(img, str(score), (center[0], pt1[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        # text = '{}:{:.1f}%'.format(class_name, score * 100)
        # txt_color = (0, 0, 0) if np.mean(np.array(color)/255) > 0.5 else (255, 255, 255)
        # font = cv2.FONT_HERSHEY_SIMPLEX

        # txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        # txt_bk_color = (np.array(color)/255 * 255 * 0.7).astype(np.uint8).tolist()
        # txt_bk_color = color
        # cv2.rectangle(
        #     img,
        #     (pt1[0], pt1[1] - 1),
        #     (pt1[0] + txt_size[0] + 1, pt1[1] - int(1.5*txt_size[1])),
        #     txt_bk_color,
        #     -1
        # )
        # cv2.putText(img, text, (pt1[0], pt1[1] - int(0.5 * txt_size[1])), font, 0.4, txt_color, thickness=1)
    return img

def ev_repr_to_img(x):
    # x = x.cpu().numpy()
    x = x[0]
    ch, ht, wd = x.shape[-3:]
    assert ch > 1 and ch % 2 == 0
    ev_repr_reshaped = rearrange(x, '(posneg C) H W -> posneg C H W', posneg=2)
    img_neg = np.asarray(reduce(ev_repr_reshaped[0], 'C H W -> H W', 'sum'), dtype='int32')
    img_pos = np.asarray(reduce(ev_repr_reshaped[1], 'C H W -> H W', 'sum'), dtype='int32')
    img_diff = img_pos - img_neg
    # img = 127 * np.ones((ht, wd, 3), dtype=np.uint8)
    # img[img_diff > 0] = 255
    # img[img_diff < 0] = 0
    img = 255 * np.ones((ht, wd, 3), dtype=np.uint8)
    img[img_diff > 0] = (0, 0, 230)
    img[img_diff < 0] = (230, 0, 0)
    return img
class Module(pl.LightningModule):
    def __init__(self, full_config: DictConfig):
        super().__init__()

        self.full_config = full_config

        self.mdl_config = full_config.model
        in_res_hw = tuple(self.mdl_config.backbone.in_res_hw)
        self.input_padder = InputPadderFromShape(desired_hw=in_res_hw)

        self.mdl = YoloXDetector(self.mdl_config)

        self.mode_2_rnn_states: Dict[Mode, RNNStates] = {
            Mode.TRAIN: RNNStates(),
            Mode.VAL: RNNStates(),
            Mode.TEST: RNNStates(),
        }

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_name = self.full_config.dataset.name
        self.mode_2_hw: Dict[Mode, Optional[Tuple[int, int]]] = {}
        self.mode_2_batch_size: Dict[Mode, Optional[int]] = {}
        self.mode_2_psee_evaluator: Dict[Mode, Optional[PropheseeEvaluator]] = {}
        self.mode_2_sampling_mode: Dict[Mode, DatasetSamplingMode] = {}

        self.started_training = True

        dataset_train_sampling = self.full_config.dataset.train.sampling
        dataset_eval_sampling = self.full_config.dataset.eval.sampling
        assert dataset_train_sampling in iter(DatasetSamplingMode)
        assert dataset_eval_sampling in (DatasetSamplingMode.STREAM, DatasetSamplingMode.RANDOM)
        if stage == 'fit':  # train + val
            self.train_config = self.full_config.training
            self.train_metrics_config = self.full_config.logging.train.metrics

            if self.train_metrics_config.compute:
                self.mode_2_psee_evaluator[Mode.TRAIN] = PropheseeEvaluator(
                    dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_psee_evaluator[Mode.VAL] = PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.TRAIN] = dataset_train_sampling
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling

            for mode in (Mode.TRAIN, Mode.VAL):
                self.mode_2_hw[mode] = None
                self.mode_2_batch_size[mode] = None
            self.started_training = False
        elif stage == 'validate':
            mode = Mode.VAL
            self.mode_2_psee_evaluator[mode] = PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        elif stage == 'test':
            mode = Mode.TEST
            self.mode_2_psee_evaluator[mode] = PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.TEST] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        else:
            raise NotImplementedError

    # def forward(self,
    #             event_tensor: th.Tensor,
    #             previous_states: Optional[MambaStates] = None,
    #             retrieve_detections: bool = True,
    #             targets=None) \
    #         -> Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None], MambaStates]:
    #     return self.mdl(x=event_tensor,
    #                     previous_states=previous_states,
    #                     retrieve_detections=retrieve_detections,
    #                     targets=targets)
    def forward(self,
                event_tensor: th.Tensor,
                previous_states: Optional[LstmStates] = None) \
            -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        output = self.mdl.forward_backbone(x=event_tensor,
                        previous_states=previous_states)[0]
        output = [output[i] for i in [1, 2, 3, 4]]
        return output
    def get_worker_id_from_batch(self, batch: Any) -> int:
        return batch['worker_id']

    def get_data_from_batch(self, batch: Any):
        return batch['data']

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        batch = merge_mixed_batches(batch)
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        mode = Mode.TRAIN
        self.started_training = True
        step = self.trainer.global_step
        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        token_mask_sequence = data.get(DataType.TOKEN_MASK, None)

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        backbone_feature_selector = BackboneFeatureSelector()
        ev_repr_selector = EventReprSelector()
        obj_labels = list()
        for tidx in range(sequence_len):
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            if token_mask_sequence is not None:
                token_masks = self.input_padder.pad_token_mask(token_mask=token_mask_sequence[tidx])
            else:
                token_masks = None

            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            backbone_features, states = self.mdl.forward_backbone(x=ev_tensors,
                                                                  previous_states=prev_states,
                                                                  token_mask=token_masks)
            prev_states = states

            current_labels, valid_batch_indices = sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
            # Store backbone features that correspond to the available labels.
            if len(current_labels) > 0:
                backbone_feature_selector.add_backbone_features(backbone_features=backbone_features,
                                                                selected_indices=valid_batch_indices)
                obj_labels.extend(current_labels)
                ev_repr_selector.add_event_representations(event_representations=ev_tensors,
                                                           selected_indices=valid_batch_indices)

        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)
        assert len(obj_labels) > 0
        # Batch the backbone features and labels to parallelize the detection code.
        selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()
        labels_yolox = ObjectLabels.get_labels_as_batched_tensor(obj_label_list=obj_labels, format_='yolox')
        labels_yolox = labels_yolox.to(dtype=self.dtype)

        predictions, losses = self.mdl.forward_detect(backbone_features=selected_backbone_features,
                                                      targets=labels_yolox)

        if self.mode_2_sampling_mode[mode] in (DatasetSamplingMode.MIXED, DatasetSamplingMode.RANDOM):
            # We only want to evaluate the last batch_size samples if we use random sampling (or mixed).
            # This is because otherwise we would mostly evaluate the init phase of the sequence.
            predictions = predictions[-batch_size:]
            obj_labels = obj_labels[-batch_size:]

        pred_processed = postprocess(prediction=predictions,
                                     num_classes=self.mdl_config.head.num_classes,
                                     conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                     nms_thre=self.mdl_config.postprocess.nms_threshold)

        loaded_labels_proph, yolox_preds_proph = to_prophesee(obj_labels, pred_processed)

        assert losses is not None
        assert 'loss' in losses

        # For visualization, we only use the last batch_size items.
        output = {
            ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-batch_size:],
            ObjDetOutput.PRED_PROPH: yolox_preds_proph[-batch_size:],
            ObjDetOutput.EV_REPR: ev_repr_selector.get_event_representations_as_list(start_idx=-batch_size),
            ObjDetOutput.SKIP_VIZ: False,
            'loss': losses['loss']
        }

        # Logging
        prefix = f'{mode_2_string[mode]}/'
        log_dict = {f'{prefix}{k}': v for k, v in losses.items()}
        self.log_dict(log_dict, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)

        if mode in self.mode_2_psee_evaluator:
            self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
            self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)
            if self.train_metrics_config.detection_metrics_every_n_steps is not None and \
                    step > 0 and step % self.train_metrics_config.detection_metrics_every_n_steps == 0:
                self.run_psee_evaluator(mode=mode)

        return output

    def _val_test_step_impl(self, batch: Any, mode: Mode, batch_idx: Any) -> Optional[STEP_OUTPUT]:
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        assert mode in (Mode.VAL, Mode.TEST)
        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        backbone_feature_selector = BackboneFeatureSelector()
        ev_repr_selector = EventReprSelector()
        obj_labels = list()
        for tidx in range(sequence_len):
            collect_predictions = (tidx == sequence_len - 1) or \
                                  (self.mode_2_sampling_mode[mode] == DatasetSamplingMode.STREAM)
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            # current_labels2 = None
            # loaded_label_list_proph = None
            # if collect_predictions:
            #     current_labels2, valid_batch_indices2 = sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
            #     # print(len(current_labels2))
            #     if len(current_labels2) > 0:
            #         loaded_label_list_proph = []
            #         for loaded_labels in current_labels2:
            #             loaded_labels.numpy_()
            #             loaded_label_proph = np.zeros((len(loaded_labels),), dtype=BBOX_DTYPE)
            #             for name in BBOX_DTYPE.names:
            #                 if name == 'track_id':
            #                     # We don't have that and don't need it
            #                     continue
            #                 loaded_label_proph[name] = np.asarray(loaded_labels.get(name), dtype=BBOX_DTYPE[name])
            #             # print(loaded_label_proph)
            #             loaded_label_list_proph.append(loaded_label_proph)

            # backbone_features, states = self.mdl.forward_backbone(x=ev_tensors, previous_states=prev_states, batch_idx=batch_idx, current_labels=loaded_label_list_proph)
            backbone_features, states = self.mdl.forward_backbone(x=ev_tensors, previous_states=prev_states)
            prev_states = states

            if collect_predictions:
                current_labels, valid_batch_indices = sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
                # print(current_labels)
                # Store backbone features that correspond to the available labels.
                if len(current_labels) > 0:
                    backbone_feature_selector.add_backbone_features(backbone_features=backbone_features,
                                                                    selected_indices=valid_batch_indices)

                    obj_labels.extend(current_labels)
                    ev_repr_selector.add_event_representations(event_representations=ev_tensors,
                                                               selected_indices=valid_batch_indices)
        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)
        if len(obj_labels) == 0:
            return {ObjDetOutput.SKIP_VIZ: True}
        selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()
        # print(selected_backbone_features.keys())
        predictions, _ = self.mdl.forward_detect(backbone_features=selected_backbone_features)

        pred_processed = postprocess(prediction=predictions,
                                     num_classes=self.mdl_config.head.num_classes,
                                     conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                     nms_thre=self.mdl_config.postprocess.nms_threshold)

        loaded_labels_proph, yolox_preds_proph = to_prophesee(obj_labels, pred_processed)

        # For visualization, we only use the last item (per batch).
        output = {
            ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-1],
            ObjDetOutput.PRED_PROPH: yolox_preds_proph[-1],
            ObjDetOutput.EV_REPR: ev_repr_selector.get_event_representations_as_list(start_idx=-1)[0],
            ObjDetOutput.SKIP_VIZ: False,
        }

        events_img = ev_repr_to_img(ev_tensors.cpu().numpy())
        events_img1 = events_img.copy()
        predictions_proph = output[ObjDetOutput.PRED_PROPH]
        img = draw_bboxes2(events_img1, predictions_proph)
        save_root = os.path.join('./vis_det_gen4_smamba', )
        os.makedirs(save_root, exist_ok=True)
        cv2.imwrite(os.path.join(save_root, str(batch_idx)+".png"), img)

        gt_proph = output[ObjDetOutput.LABELS_PROPH]
        # print(gt_proph.shape)
        img_gt = draw_bboxes2(events_img, gt_proph)
        cv2.imwrite(os.path.join(save_root, str(batch_idx)+"_GT.png"), img_gt)


        if self.started_training:
            self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
            self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)

        return output

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode=Mode.VAL, batch_idx=batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        # print(batch_idx)
        return self._val_test_step_impl(batch=batch, mode=Mode.TEST, batch_idx=batch_idx)

    def run_psee_evaluator(self, mode: Mode):
        psee_evaluator = self.mode_2_psee_evaluator[mode]
        batch_size = self.mode_2_batch_size[mode]
        hw_tuple = self.mode_2_hw[mode]
        if psee_evaluator is None:
            warn(f'psee_evaluator is None in {mode=}', UserWarning, stacklevel=2)
            return
        assert batch_size is not None
        assert hw_tuple is not None
        if psee_evaluator.has_data():
            metrics = psee_evaluator.evaluate_buffer(img_height=hw_tuple[0],
                                                     img_width=hw_tuple[1])
            assert metrics is not None

            prefix = f'{mode_2_string[mode]}/'
            step = self.trainer.global_step
            log_dict = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    value = torch.tensor(v)
                elif isinstance(v, np.ndarray):
                    value = torch.from_numpy(v)
                elif isinstance(v, torch.Tensor):
                    value = v
                else:
                    raise NotImplementedError
                assert value.ndim == 0, f'tensor must be a scalar.\n{v=}\n{type(v)=}\n{value=}\n{type(value)=}'
                # put them on the current device to avoid this error: https://github.com/Lightning-AI/lightning/discussions/2529
                log_dict[f'{prefix}{k}'] = value.to(self.device)
            # Somehow self.log does not work when we eval during the training epoch.
            self.log_dict(log_dict, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            if dist.is_available() and dist.is_initialized():
                # We now have to manually sync (average the metrics) across processes in case of distributed training.
                # NOTE: This is necessary to ensure that we have the same numbers for the checkpoint metric (metadata)
                # and wandb metric:
                # - checkpoint callback is using the self.log function which uses global sync (avg across ranks)
                # - wandb uses log_metrics that we reduce manually to global rank 0
                dist.barrier()
                for k, v in log_dict.items():
                    dist.reduce(log_dict[k], dst=0, op=dist.ReduceOp.SUM)
                    if dist.get_rank() == 0:
                        log_dict[k] /= dist.get_world_size()
            if self.trainer.is_global_zero:
                # For some reason we need to increase the step by 2 to enable consistent logging in wandb here.
                # I might not understand wandb login correctly. This works reasonably well for now.
                add_hack = 2
                self.logger.log_metrics(metrics=log_dict, step=step + add_hack)

            psee_evaluator.reset_buffer()
        else:
            warn(f'psee_evaluator has not data in {mode=}', UserWarning, stacklevel=2)

    def on_train_epoch_end(self) -> None:
        mode = Mode.TRAIN
        if mode in self.mode_2_psee_evaluator and \
                self.train_metrics_config.detection_metrics_every_n_steps is None and \
                self.mode_2_hw[mode] is not None:
            # For some reason PL calls this function when resuming.
            # We don't know yet the value of train_height_width, so we skip this
            self.run_psee_evaluator(mode=mode)

    def on_validation_epoch_end(self) -> None:
        mode = Mode.VAL
        if self.started_training:
            assert self.mode_2_psee_evaluator[mode].has_data()
            self.run_psee_evaluator(mode=mode)

    def on_test_epoch_end(self) -> None:
        mode = Mode.TEST
        assert self.mode_2_psee_evaluator[mode].has_data()
        self.run_psee_evaluator(mode=mode)

    def configure_optimizers(self) -> Any:
        lr = self.train_config.learning_rate
        weight_decay = self.train_config.weight_decay
        optimizer = th.optim.AdamW(self.mdl.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_params = self.train_config.lr_scheduler
        if not scheduler_params.use:
            return optimizer

        total_steps = scheduler_params.total_steps
        assert total_steps is not None
        assert total_steps > 0
        # Here we interpret the final lr as max_lr/final_div_factor.
        # Note that Pytorch OneCycleLR interprets it as initial_lr/final_div_factor:
        final_div_factor_pytorch = scheduler_params.final_div_factor / scheduler_params.div_factor
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            div_factor=scheduler_params.div_factor,
            final_div_factor=final_div_factor_pytorch,
            total_steps=total_steps,
            pct_start=scheduler_params.pct_start,
            cycle_momentum=False,
            anneal_strategy='linear')
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": 'learning_rate',
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
