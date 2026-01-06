import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def statistical_analysis(v):
    a = np.zeros((5))
    a[0] = np.min(v)
    a[1] = np.max(v)
    a[2] = np.mean(v)
    a[3] = np.median(v)
    a[4] = np.std(v)
    return a


def plot_Results():
    for a in range(1):
        Eval =np.load('Evaluate_all.npy',allow_pickle=True)[a]

        Terms = ['Scalability (%)','Security (%)','Serving Delay (ms)', 'Time(s)', 'Energy Consumption (J)', 'Throughput (bits per s)', 'Overhead Probability', 'Resource Utilization (%)', 'Cost (doller']
        for b in range(len(Terms)):
            learnper = [1, 2, 3, 4, 5]

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Eval[:, 0,b], color='#aaff32', width=0.10, label="NRO")
            ax.bar(X + 0.10, Eval[:, 1,b], color='#ad03de', width=0.10, label="FLO")
            ax.bar(X + 0.20, Eval[:, 2,b], color='#8c564b', width=0.10, label="GOA")
            ax.bar(X + 0.30, Eval[:, 3,b], color='#ff000d', width=0.10, label="BWK")
            ax.bar(X + 0.40, Eval[:, 4,b], color='k', width=0.10, label="IRP-BWK")
            # plt.xticks(X + 0.25, ('5', '10', '15', '20', '25'))


            labels = ['50', '100', '150', '200', '250']
            plt.xticks(X+0.2, labels)
            plt.xlabel('No of Task')
            plt.ylabel(Terms[b])

            plt.tight_layout()
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_%s_bar.png" % (a + 1, Terms[b])
            plt.savefig(path1)
            plt.show()
def plot_Fitness():
    Task = [50, 100, 150, 200, 250]
    for a in range(5):
        Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Algorithm = ['NRO', 'FLO', 'GOA', 'BWK', 'IRP-BWK']

        conv = np.load('Fitness.npy', allow_pickle=True)[a]
        ind = np.argsort(conv[:, conv.shape[1] - 1])
        x = conv[ind[0], :].copy()
        y = conv[4, :].copy()
        conv[4, :] = x
        conv[ind[0], :] = y

        Value = np.zeros((conv.shape[0], 5))
        for j in range(conv.shape[0]):
            Value[j, 0] = np.min(conv[j, :])
            Value[j, 1] = np.max(conv[j, :])
            Value[j, 2] = np.mean(conv[j, :])
            Value[j, 3] = np.median(conv[j, :])
            Value[j, 4] = np.std(conv[j, :])

        Table = PrettyTable()
        Table.add_column("ALGORITHMS", Statistics)
        for j in range(len(Algorithm)):
            Table.add_column(Algorithm[j], Value[j, :])
        print('--------------------------------------------------Task - ', Task[a], ' - Statistical Analysis--------------------------------------------------')
        print(Table)

        iteration = np.arange(conv.shape[1])
        plt.plot(iteration, conv[0, :], color='#7ebd01', linewidth=3, marker='>', markerfacecolor='blue', markersize=12,
                 label="NRO")
        plt.plot(iteration, conv[1, :], color='#ef4026', linewidth=3, marker='>', markerfacecolor='red', markersize=12,
                 label="FLO")
        plt.plot(iteration, conv[2, :], color='#12e193', linewidth=3, marker='>', markerfacecolor='green', markersize=12,
                 label="GOA")
        plt.plot(iteration, conv[3, :], color='#ff0490', linewidth=3, marker='>', markerfacecolor='yellow',
                 markersize=12,
                 label="BWK")
        plt.plot(iteration, conv[4, :], color='k', linewidth=3, marker='>', markerfacecolor='cyan', markersize=12,
                 label="IRP-BWK")
        plt.xlabel('No. of Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.tight_layout()
        path1 = "./Results/convergence_%s.jpg"  %(str(a+1))
        plt.savefig(path1)
        plt.show()


if __name__ == '__main__':
    # plot_Results()
    plot_Fitness()