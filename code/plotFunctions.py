import numpy as np
import  matplotlib.pyplot as plt


class plotGridSearchResult:
    def __init__(self, title, xLabel, yLabel, metricVals, results, isLogxScale=False):
        plt.figure(figsize=(7, 7), dpi=180)
        plt.title(title,
          fontsize=16)

        plt.xlabel(xLabel)#("Epochs")
        plt.ylabel(yLabel)#"Score")
        plt.grid()
        ax = plt.axes()
        #ax.set_ylim(0.73, 1) # this will be done later

        # ax.set_ylim(0.73, 1)       
        #print(results.keys)
        
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, "score")]
            sample_score_std = results['std_%s_%s' % (sample, "score")]
            ax.fill_between(metricVals, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color='g')

            if isLogxScale:               
                ax.semilogx(metricVals, sample_score_mean, style, color='g',
                        alpha=1 if sample == 'test' else 0.7,
                        label="%s (%s)" % ("Accuracy", sample))
            else:               
                ax.plot(metricVals, sample_score_mean, style, color='g',
                        alpha=1 if sample == 'test' else 0.7,
                        label="%s (%s)" % ("Accuracy", sample))

        best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]
        best_score = results['mean_test_%s' % "score"][best_index]

        ax.set_ylim(ax.get_ylim())

        # # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([metricVals[best_index], ] * 2, [0, best_score],
                  linestyle='-.', color='g', marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("Best CVScore\n(%d, %0.2f%%)" % (metricVals[best_index], best_score*100),
                      (metricVals[best_index], best_score + 0.005))

        plt.legend(loc="best")
        plt.grid('off')
        plt.show()

