import torch

from batch_bald import exact as joint_entropy_exact, torch_utils, sampling as joint_entropy_sampling
import math

from batch_bald.acquisition_functions import AcquisitionFunction
from batch_bald.reduced_consistent_mc_sampler import reduced_eval_consistent_bayesian_model


compute_multi_bald_bag_multi_bald_batch_size = None


def compute_multi_bald_batch(
    bayesian_model,
    unlabeled_loader,
    num_classes,
    k,
    b,
):
    partial_multi_bald_B, logits_B_K_C = reduced_eval_consistent_bayesian_model(
        bayesian_model=bayesian_model,
        acquisition_function=AcquisitionFunction.bald,
        num_classes=num_classes,
        k=k,
        unlabeled_loader=unlabeled_loader,
    )

    # Now we can compute the conditional entropy
    conditional_entropies_B = joint_entropy_exact.batch_conditional_entropy_B(logits_B_K_C)

    # We turn the logits into probabilities.
    probs_B_K_C = logits_B_K_C.exp_()

    torch_utils.gc_cuda()

    with torch.no_grad():
        num_samples_per_ws = 40000 // k
        num_samples = num_samples_per_ws * k

        # # KC_memory = k*num_classes*8
        # sample_MK_memory = num_samples * k * 8
        # MC_memory = num_samples * num_classes * 8
        # copy_buffer_memory = 256 * num_samples * num_classes * 8
        # slack_memory = 2 * 2 ** 30
        # multi_bald_batch_size = (
        #     torch_utils.get_cuda_available_memory() - (sample_MK_memory + copy_buffer_memory + slack_memory)
        # ) // MC_memory
        #
        # global compute_multi_bald_bag_multi_bald_batch_size
        # if compute_multi_bald_bag_multi_bald_batch_size != multi_bald_batch_size:
        #     compute_multi_bald_bag_multi_bald_batch_size = multi_bald_batch_size

        multi_bald_batch_size = 16

        subset_acquisition_bag = []
        acquisition_bag_scores = []

        # We use this for early-out in the b==0 case.
        MIN_SPREAD = 0.1

        if b == 0:
            b = 100
            early_out = True
        else:
            early_out = False

        prev_joint_probs_M_K = None
        prev_samples_M_K = None
        for i in range(b):
            torch_utils.gc_cuda()

            if i > 0:
                # Compute the joint entropy
                joint_entropies_B = torch.empty((len(probs_B_K_C),), dtype=torch.float64)

                exact_samples = num_classes ** i
                if exact_samples <= num_samples:
                    # prev_joint_probs_M_K = joint_entropy_exact.joint_probs_M_K(
                    #     probs_B_K_C[subset_acquisition_bag[-1]][None].cuda(),
                    #     prev_joint_probs_M_K=prev_joint_probs_M_K,
                    # )
                    prev_joint_probs_M_K = joint_entropy_exact.joint_probs_M_K(
                        probs_B_K_C[subset_acquisition_bag[-1]][None],
                        prev_joint_probs_M_K=prev_joint_probs_M_K,
                    )

                    # torch_utils.cuda_meminfo()
                    batch_exact_joint_entropy(
                        probs_B_K_C, prev_joint_probs_M_K, multi_bald_batch_size, joint_entropies_B
                    )
                else:
                    if prev_joint_probs_M_K is not None:
                        prev_joint_probs_M_K = None
                        torch_utils.gc_cuda()

                    # Gather new traces for the new subset_acquisition_bag.
                    prev_samples_M_K = joint_entropy_sampling.sample_M_K(
                        probs_B_K_C[subset_acquisition_bag], S=num_samples_per_ws
                    )
                    # prev_samples_M_K = joint_entropy_sampling.sample_M_K(
                    #     probs_B_K_C[subset_acquisition_bag].cuda(), S=num_samples_per_ws
                    # )

                    # torch_utils.cuda_meminfo()
                    for joint_entropies_b, probs_b_K_C in torch_utils.split_tensors(joint_entropies_B, probs_B_K_C, multi_bald_batch_size):
                        joint_entropies_b.copy_(
                            joint_entropy_sampling.batch(probs_b_K_C, prev_samples_M_K), non_blocking=True
                        )
                    # for joint_entropies_b, probs_b_K_C in torch_utils.split_tensors(joint_entropies_B, probs_B_K_C, multi_bald_batch_size):
                    #     joint_entropies_b.copy_(
                    #         joint_entropy_sampling.batch(probs_b_K_C.cuda(), prev_samples_M_K), non_blocking=True
                    #     )

                        # torch_utils.cuda_meminfo()

                    prev_samples_M_K = None
                    torch_utils.gc_cuda()

                partial_multi_bald_B = joint_entropies_B - conditional_entropies_B
                joint_entropies_B = None

            # Don't allow reselection
            partial_multi_bald_B[subset_acquisition_bag] = -math.inf

            winner_index = partial_multi_bald_B.argmax().item()

            # Actual MultiBALD is:
            actual_multi_bald_B = partial_multi_bald_B[winner_index] - torch.sum(
                conditional_entropies_B[subset_acquisition_bag]
            )
            actual_multi_bald_B = actual_multi_bald_B.item()

            # If we early out, we don't take the point that triggers the early out.
            # Only allow early-out after acquiring at least 1 sample.
            if early_out and i > 1:
                current_spread = actual_multi_bald_B[winner_index] - actual_multi_bald_B.median()
                if current_spread < MIN_SPREAD:
                    break

            acquisition_bag_scores.append(actual_multi_bald_B)
            subset_acquisition_bag.append(winner_index)

    return subset_acquisition_bag, acquisition_bag_scores


def batch_exact_joint_entropy(probs_B_K_C, prev_joint_probs_M_K, chunk_size, out_joint_entropies_B):
    """This one switches between devices, too."""
    for joint_entropies_b, probs_b_K_C in torch_utils.split_tensors(out_joint_entropies_B, probs_B_K_C, chunk_size):
        joint_entropies_b.copy_(
            joint_entropy_exact.batch(probs_b_K_C, prev_joint_probs_M_K), non_blocking=True
        )
    # for joint_entropies_b, probs_b_K_C in torch_utils.split_tensors(out_joint_entropies_B, probs_B_K_C, chunk_size):
    #     joint_entropies_b.copy_(
    #         joint_entropy_exact.batch(probs_b_K_C.cuda(), prev_joint_probs_M_K), non_blocking=True
    #     )

    return joint_entropies_b


def batch_exact_joint_entropy_logits(logits_B_K_C, prev_joint_probs_M_K, chunk_size, out_joint_entropies_B):
    """This one switches between devices, too."""
    # for joint_entropies_b, logits_b_K_C in torch_utils.split_tensors(out_joint_entropies_B, logits_B_K_C, chunk_size):
    #     joint_entropies_b.copy_(
    #         joint_entropy_exact.batch(logits_b_K_C.cuda().exp(), prev_joint_probs_M_K), non_blocking=True
    #     )
    for joint_entropies_b, logits_b_K_C in torch_utils.split_tensors(out_joint_entropies_B, logits_B_K_C, chunk_size):
        joint_entropies_b.copy_(
            joint_entropy_exact.batch(logits_b_K_C.exp(), prev_joint_probs_M_K), non_blocking=True
        )

    return joint_entropies_b
