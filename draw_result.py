import matplotlib.pylab as plt
import matplotlib.cm as cm
import numpy as np
import utils
import itertools


def draw_gene_error(test_error,criteria,initial_sample_size,batch_size,color,fontsize=24,isLegend=True):
    plt.figure(1,[8,5])
    plt.clf()
    sample_size_list = np.arange(batch_size+initial_sample_size,batch_size*(test_error.shape[0]+1)+initial_sample_size,batch_size)
    plt.plot(sample_size_list, test_error, c='k', label="Generalization_error")
    for criterion in criteria:
        stopping_data_size = batch_size+initial_sample_size+criterion.stop_timings*batch_size
        plt.plot([stopping_data_size, stopping_data_size], [test_error.min(), test_error.max()], c=color[criterion.criterion_name],
                 linestyle="dashed", label=r"$\lambda={0}$".format(criterion.threshold))
    if isLegend:
        plt.legend(fontsize=fontsize)
    plt.xlabel("Sample size",fontsize=fontsize)
    plt.ylabel("Generalization error",fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.pause(0.01)


def draw_correlations(result,y_label,data_names,color,label,fontsize=24,ylim=1):
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


def draw_epsilon(criteria,initial_sample_size,batch_size,color,fontsize=24,isLegend=True):
    plt.figure(3,[8,5])
    plt.clf()
    plt.ylim(-0.1, 1.1)
    for criterion in criteria:
        sample_size_list = np.arange(batch_size + initial_sample_size,
                                                   batch_size*(criterion.error_ratio.shape[0]+1) + initial_sample_size,
                                                   batch_size)
        stopping_data_size = batch_size + initial_sample_size + criterion.stop_timings * batch_size
        plt.plot(sample_size_list, criterion.error_ratio, c="k")
        plt.plot([batch_size+initial_sample_size, batch_size*criterion.error_ratio.shape[0] + initial_sample_size], [criterion.threshold, criterion.threshold], c=color[criterion.criterion_name],label=r"$\lambda={0}$".format(criterion.threshold))
        plt.plot([stopping_data_size, stopping_data_size], [0, criterion.error_ratio.max()], c=color[criterion.criterion_name],linestyle="dashed")
    if isLegend:
        plt.legend(fontsize=fontsize)
    plt.xlabel("Sample size",fontsize=fontsize)
    plt.ylabel("Error ratio",fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.pause(0.01)


def draw_correlation(test_error,criterion,label,color="k",fontsize=24,isLegend=True):
    plt.figure(4,[8,8])
    plt.clf()
    indecies = utils.calc_min_list(criterion)
    plt.scatter(test_error[indecies],criterion[indecies],c=color,label=label)
    if isLegend:
        plt.legend(fontsize=fontsize)
    plt.xlabel("Generalization error",fontsize=fontsize)
    plt.ylabel("Error ratio",fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.tight_layout()
    plt.pause(0.01)


def set_axes():
    plt.figure(4,[15,9])
    axes = []
    for i in range(9):
        axes.append(plt.subplot(3,3,i+1))
    return axes


