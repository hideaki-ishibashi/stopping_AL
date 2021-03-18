import torch
from batch_bald import torch_utils
import enum


def bald_acquisition_function(logits_b_K_C):
    return torch_utils.mutual_information(logits_b_K_C)


class AcquisitionFunction(enum.Enum):

    bald = "bald"

    @property
    def scorer(self):
        return bald_acquisition_function

    def compute_scores(self, logits_B_K_C):
        scorer = self.scorer
        B, K, C = logits_B_K_C.shape

        with torch.no_grad():
            scores_B = torch.empty((B,), dtype=torch.float64)
            # scores_B = torch.empty((B,), dtype=torch.float64).cuda()

            # torch_utils.gc_cuda()
            # KC_memory = K * C * 8
            # batch_size = min(torch_utils.get_cuda_available_memory() // KC_memory, 8192)
            batch_size = 4096

            # if device.type == "cuda":
            #     torch_utils.gc_cuda()
            #     KC_memory = K * C * 8
            #     batch_size = min(torch_utils.get_cuda_available_memory() // KC_memory, 8192)
            # else:
            #     batch_size = 4096

            # for scores_b, logits_b_K_C in torch_utils.split_tensors(scores_B, logits_B_K_C, batch_size):
            #     scores_b.copy_(scorer(logits_b_K_C.cuda()), non_blocking=True)
            for scores_b, logits_b_K_C in torch_utils.split_tensors(scores_B, logits_B_K_C, batch_size):
                scores_b.copy_(scorer(logits_b_K_C), non_blocking=True)

        return scores_B
