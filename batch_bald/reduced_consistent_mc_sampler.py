import torch
import torch.nn.functional as F


def reduced_eval_consistent_bayesian_model(
    bayesian_model,
    acquisition_function,
    num_classes,
    k,
    unlabeled_loader,
):

    # We start with all data in the acquired data.
    logits_B_K_C = []

    bayesian_model.eval()

    for x, _ in unlabeled_loader:
        x = x
        # x = x.cuda()
        with torch.no_grad():
            logits = bayesian_model(x, k=k)
            # B K C by permute function
            logits = F.log_softmax(logits, dim=-1)
            logits_B_K_C.append(logits.cpu())
    logits_B_K_C = torch.cat(logits_B_K_C, 0)
    scores_B = acquisition_function.compute_scores(logits_B_K_C)

    return scores_B, logits_B_K_C
