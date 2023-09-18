import math


def adjust_weight(epoch, start_epoch=10, total_epochs=40, base_weight=0.5, max_weight=0.9, ITERS_PER_EPOCH=1, iters=-1):
    if epoch < start_epoch:
        return base_weight
    if epoch >= total_epochs:
        return max_weight
    if iters == -1:
        iters = epoch * ITERS_PER_EPOCH
    total_iters = ITERS_PER_EPOCH * (total_epochs - start_epoch)
    iters = iters - ITERS_PER_EPOCH * start_epoch
    # keep_rate = base_weight + (max_weight - base_weight) \
    #     * (math.cos(iters / total_iters * math.pi) + 1) * 0.5  # 0.9 --> 0.5
    keep_rate = max_weight - (max_weight - base_weight) \
        * (math.cos(iters / total_iters * math.pi) + 1) * 0.5  # 0.5 --> 0.9

    return keep_rate
