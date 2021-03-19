import matplotlib.pylab as plt
import matplotlib.cm as cm
import numpy as np
import utils
import itertools


def draw_gene_error(test_error,criteria,color,fontsize=24,isLegend=True):
    plt.figure(1,[8,5])
    plt.clf()
    plt.plot(range(1,test_error.shape[0]+1), test_error, c='k', label="generalization_error")
    for criterion in criteria:
        plt.plot([criterion.stop_timings, criterion.stop_timings], [test_error.min(), test_error.max()], c=color[criterion.criterion_name],
                 linestyle="dashed", label=r"$\lambda={0}$".format(criterion.threshold))
    if isLegend:
        plt.legend(fontsize=fontsize)
    plt.xlabel("data size",fontsize=fontsize)
    plt.ylabel("generalization error",fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.pause(0.01)
# def draw_gene_error(test_error,stopping_timings,color,thresholds,fontsize=24):
#     plt.figure(1,[8,5])
#     plt.clf()
#     plt.plot(range(1,test_error.shape[0]+1), test_error, c='k', label="generalization_error")
#     for i,stopping_timing in enumerate(stopping_timings):
#         plt.plot([stopping_timing, stopping_timing], [test_error.min(), test_error.max()], c=color[i],
#              linestyle="dashed", label=r"$\lambda={0}$".format(thresholds[i]))
#     plt.legend(fontsize=fontsize)
#     plt.xlabel("data size",fontsize=fontsize)
#     plt.ylabel("generalization error",fontsize=fontsize)
#     plt.tick_params(labelsize=fontsize)
#     plt.pause(0.01)


def draw_result_list(result,y_label,data_names,color,label,fontsize=24,ylim=1):
    plt.figure(4,[25,10])
    plt.clf()
    plt.ylim(0, ylim)
    y_ticks = np.linspace(0,ylim,11)
    sum_left = 0
    lefts = []
    for i,data_name_list in enumerate(data_names):
        left = np.arange(len(data_name_list))+sum_left
        plt.bar(left, result[i], color=color[i], label=label[i],align="center" , ecolor="k")
        sum_left += len(data_name_list)
        lefts.append(left)
    left = np.concatenate(lefts)
    data_names = list(itertools.chain.from_iterable(data_names))
    plt.xticks(left, data_names,rotation=10)
    plt.yticks(y_ticks)
    plt.ylabel(y_label,fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.legend(fontsize=fontsize, loc='lower right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='dotted', color='k')
    plt.pause(0.01)


def draw_epsilon(criteria,color,fontsize=24,isLegend=True):
    plt.figure(3,[8,5])
    plt.clf()
    plt.ylim(-0.1, 1.1)
    for criterion in criteria:
        plt.plot(range(1,criterion.criterion.shape[0]+1), criterion.criterion, c="k")
        plt.plot([1, criterion.criterion.shape[0]+1], [criterion.threshold, criterion.threshold], c=color[criterion.criterion_name],label=r"$\lambda={0}$".format(criterion.threshold))
        plt.plot([criterion.stop_timings, criterion.stop_timings], [0, criterion.criterion.max()], c=color[criterion.criterion_name],linestyle="dashed")
    if isLegend:
        plt.legend(fontsize=fontsize)
    plt.xlabel("data size",fontsize=fontsize)
    plt.ylabel("error ratio",fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.pause(0.01)

# def draw_epsilon(criterion,color,thresholds,fontsize=24,isLegend=True):
#     plt.figure(3,[8,5])
#     plt.clf()
#     plt.ylim(-0.1, 1.1)
#     plt.plot(range(1, criterion.shape[0] + 1), criterion, c="k",label="Error ratio")
#     for i,threshold in enumerate(thresholds):
#         plt.plot([1, criterion.shape[0]+1], [threshold, threshold], c=color[i],label=r"$\lambda={0}$".format(threshold))
#         # plt.plot([criterion.stop_timings, criterion.stop_timings], [0, criterion.criterion.max()], c=color[criterion.criterion_name],linestyle="dashed")
#     if isLegend:
#         plt.legend(fontsize=fontsize)
#     plt.xlabel("data size",fontsize=fontsize)
#     plt.ylabel("error ratio",fontsize=fontsize)
#     plt.tick_params(labelsize=fontsize)
#     plt.pause(0.01)


def draw_correlation(test_error,criterion,label,color="k",fontsize=24,isLegend=True):
    plt.figure(4,[8,8])
    plt.clf()
    indecies = utils.calc_min_list(criterion)
    plt.scatter(test_error[indecies],criterion[indecies],c=color,label=label)
    if isLegend:
        plt.legend(fontsize=fontsize)
    plt.xlabel("generalization error",fontsize=fontsize)
    plt.ylabel("error ratio",fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.tight_layout()
    plt.pause(0.01)


def set_axes():
    plt.figure(4,[15,9])
    axes = []
    for i in range(9):
        axes.append(plt.subplot(3,3,i+1))
    return axes


