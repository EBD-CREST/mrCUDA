import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.font_manager
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'custom'

import numpy as np

import csv
import argparse
import math

COLOR = ['b', 'g', 'r', 'c', 'm', 'y',]

def parseargs():
    """
    Manage the program arguments.
    """
    parser = argparse.ArgumentParser(
        description = 'mrCUDA overhead benchmark result plotter'
    )
    parser.add_argument('type',
        choices = ('memsync', 'memsync-bw', 'mhelper-nullker', 'mhelper-memcpybw', 'record-replay',),
        help = 'Overhead type'
    )
    parser.add_argument('resultfile', type = argparse.FileType('r'),
        help = 'Result file (csv)'
    )
    return parser.parse_args()

def read_memsync_input(input_file):
    # All time is in ms.
    # All sizes are in B.
    reader = csv.DictReader(input_file, delimiter = ' ')
    result = list()
    for row in reader:
        row['total_size'] = int(row['total_size'])
        row['num_regions'] = int(row['num_regions'])
        # Filter out some results to reduce size
        if math.log(row['num_regions'], 2) % 2 == 1:
            continue
        row['memsync_time'] = float(row['memsync_time'])
        row['rcuda_time'] = float(row['rcuda_time'])
        row['local_time'] = float(row['local_time'])
        row['nvidia_time'] = float(row['nvidia_time'])
        row['other_time'] = float(row['other_time'])
        row['size_per_region'] = float(row['total_size']) / float(row['num_regions'])
        row['bw'] = row['total_size'] / row['nvidia_time'] * (10 ** -3) # MB / s
        result.append(row)
    return result

def plot_memsync(input_data):
    properties = {
        'bw_coef': 0.04721 * (10 ** 6), # 1 / s
        'bw_max': 4778.505 * (10 ** 6), # B / s
        'memsync_coef': 5.686 * (10 ** -11), # s / B
        'memsync_const': 0, # s
    }

    group_dict = dict()
    predicted_dict = dict()
    for data in input_data:
        if data['num_regions'] not in group_dict:
            group_dict[data['num_regions']] = [list(), list(),]
        group_data = group_dict[data['num_regions']]
        group_data[0].append(data['size_per_region'])
        group_data[1].append(data['local_time'] / 1000) 

        if data['num_regions'] not in predicted_dict:
            predicted_dict[data['num_regions']] = dict()
        if data['size_per_region'] not in predicted_dict[data['num_regions']]:
            predicted_dict[data['num_regions']][data['size_per_region']] = data['num_regions'] * (properties['memsync_coef'] * data['size_per_region'] + properties['memsync_const'] + data['size_per_region'] / min(properties['bw_max'], properties['bw_coef'] * data['size_per_region']))

    legend_list = list()
    i = 0
    for num_regions, group_data in sorted(group_dict.items(), key = lambda item: item[0]):
        plt.scatter(group_data[0], group_data[1], 
            c = COLOR[i % len(COLOR)],
            marker = 'o' if i < len(COLOR) else '+',
            s = 40
        )
        x, y = zip(*sorted(predicted_dict[num_regions].items(), key = lambda item: item[0]))
        p = plt.plot(x, y, COLOR[i % len(COLOR)], linewidth = 4)
        legend_list.append((p[0], '$\mathbf{2^{%d}}$ regions' % (math.log(num_regions, 2),),))
        i += 1

    p = mlines.Line2D([], [], color = 'black', linewidth = 4)
    legend_list.append((p, 'Predicted',))
    p = mlines.Line2D([], [], color = 'black', marker = 'o', markersize = 16, linestyle = 'None')
    legend_list.append((p, 'Measured',))

    legend_list.reverse()

    plt.legend(zip(*legend_list)[0], zip(*legend_list)[1],
        loc = 'upper left',
        prop = matplotlib.font_manager.FontProperties(size = 30, weight = 'bold')
    )
    plt.xscale('log', basex = 2)
    plt.yscale('log', basey = 10)
    plt.xlim(xmin = 0)
    plt.ylim(ymin = 0)

    plt.xlabel('$\mathbf{data\_size_i}$ (B)', size = 40, weight = 'bold')
    plt.ylabel('Time (s)', size = 40, weight = 'bold')

    plt.xticks(size = 35, weight = 'bold')
    plt.yticks(size = 35, weight = 'bold')

    plt.show()

def plot_memsync_bw(input_data):
    properties = {
        'bw_coef': 0.04721 * (10 ** 6), # 1 / s
        'bw_max': 4778.505 * (10 ** 6), # B / s
        'memsync_coef': 5.686 * (10 ** -11), # s / B
        'memsync_const': 0, # s
    }

    measured_data = [(row['size_per_region'], row['bw'],) for row in input_data]
    predicted_data = [(size_per_region, min(properties['bw_max'], properties['bw_coef'] * size_per_region) * (10 ** -6),) for size_per_region in sorted(set(zip(*measured_data)[0]))]

    legend_list = list()
    p = plt.scatter(
        zip(*measured_data)[0],
        zip(*measured_data)[1],
        c = COLOR[0],
        marker = 'o',
        s = 40
    )
    legend_list.append((p, 'Measured',))
    x, y = zip(*predicted_data)
    plt.plot(x, y, COLOR[0], linewidth = 4)
    p = mlines.Line2D([], [], color = COLOR[0], linewidth = 4)
    legend_list.append((p, 'Predicted',))

    plt.legend(zip(*legend_list)[0], zip(*legend_list)[1],
        loc = 'upper left',
        prop = matplotlib.font_manager.FontProperties(size = 30, weight = 'bold')
    )
    plt.xscale('log', basex = 2)
    plt.yscale('log', basey = 10)
    plt.xlim(xmin = 0)
    plt.ylim(ymin = 0)

    plt.xlabel('Size per region (B)', size = 30, weight = 'bold')
    plt.ylabel('Bandwidth (MB / s)', size = 30, weight = 'bold')

    plt.xticks(size = 25, weight = 'bold')
    plt.yticks(size = 25, weight = 'bold')

    plt.show()

def read_mhelper_input(input_file):
    # All time is in ms.
    # All sizes are in B.
    reader = csv.DictReader(input_file, delimiter = ' ')
    result = list()
    for row in reader:
        row['count'] = int(row['count'])
        row['time'] = float(row['time'])
        if 'num_calls' in row:
            row['num_calls'] = int(row['num_calls'])
        else:
            row['size_per_call'] = int(row['size_per_call'])
        result.append(row)
    return result

def plot_mhelper_nullker(input_data):
    properties = {
        'coefd': 6.87138 * (10 ** -10), #s
        'coefc': 9.98263 * (10 ** -6), # s
        'const': 0.00293373, # s
    }

    native_data = dict()
    mrcuda_data = dict()
    for data in input_data:
        if data['lib'] == 'native':
            data_dict = native_data
        else:
            data_dict = mrcuda_data
        if data['num_calls'] not in data_dict:
            data_dict[data['num_calls']] = list()
        data_dict[data['num_calls']].append(data['time'])

    x_values = list()
    y_values = list()

    for num_calls in native_data.iterkeys():
        avg_time = np.average(native_data[num_calls])
        for time in mrcuda_data[num_calls]:
            x_values.append(num_calls)
            y_values.append((time - avg_time) * (10 ** -3)) # seconds

    legend_list = list()

    p = plt.scatter(
        x_values,
        y_values,
        c = COLOR[0],
        marker = 'o',
        s = 40
    )
    legend_list.append((p, 'Measured',))

    x_values = sorted(set(x_values))
    y_values = [properties['coefc'] * x + properties['const'] for x in x_values]

    plt.plot(x_values, y_values, COLOR[0], linewidth = 4)
    p = mlines.Line2D([], [], color = COLOR[0], linewidth = 4)
    legend_list.append((p, 'Predicted',))

    plt.legend(zip(*legend_list)[0], zip(*legend_list)[1],
        loc = 'upper left',
        prop = matplotlib.font_manager.FontProperties(size = 30, weight = 'bold')
    )
    plt.xscale('log', basex = 2)
    plt.yscale('log', basey = 10)
    plt.xlim(xmin = 0)
    plt.ylim(ymin = 0)

    plt.xlabel('Number of calls', size = 30, weight = 'bold')
    plt.ylabel('Time (s)', size = 30, weight = 'bold')

    plt.xticks(size = 25, weight = 'bold')
    plt.yticks(size = 25, weight = 'bold')

    plt.show()

def plot_mhelper_memcpybw(input_data):
    properties = {
        'coefd': 6.87138 * (10 ** -10), #s
        'coefc': 9.98263 * (10 ** -6), # s
        'const': 0.00293373, # s
        'num_calls': 1000,
    }

    native_data = dict()
    mrcuda_data = dict()
    for data in input_data:
        if data['lib'] == 'native':
            data_dict = native_data
        else:
            data_dict = mrcuda_data
        if data['size_per_call'] not in data_dict:
            data_dict[data['size_per_call']] = list()
        data_dict[data['size_per_call']].append(data['time'])

    x_values = list()
    y_values = list()

    for size_per_call in native_data.iterkeys():
        avg_time = np.average(native_data[size_per_call])
        for time in mrcuda_data[size_per_call]:
            x_values.append(size_per_call)
            y_values.append((time - avg_time) * (10 ** -3)) # seconds

    legend_list = list()

    p = plt.scatter(
        x_values,
        y_values,
        c = COLOR[0],
        marker = 'o',
        s = 40
    )
    legend_list.append((p, 'Measured',))

    x_values = sorted(set(x_values))
    y_values = [properties['coefd'] * x * properties['num_calls'] + properties['coefc'] * properties['num_calls'] + properties['const'] for x in x_values]

    plt.plot(x_values, y_values, COLOR[0], linewidth = 4)
    p = mlines.Line2D([], [], color = COLOR[0], linewidth = 4)
    legend_list.append((p, 'Predicted',))

    plt.legend(zip(*legend_list)[0], zip(*legend_list)[1],
        loc = 'upper left',
        prop = matplotlib.font_manager.FontProperties(size = 40, weight = 'bold')
    )
    plt.xscale('log', basex = 2)
    plt.yscale('log', basey = 10)
    plt.xlim(xmin = 0)
    plt.ylim(ymin = 0)

    plt.xlabel('Size per calls (B)', size = 40, weight = 'bold')
    plt.ylabel('Time (s)', size = 40, weight = 'bold')

    plt.xticks(size = 35, weight = 'bold')
    plt.yticks(size = 35, weight = 'bold')

    plt.show()

def read_record_replay_input(input_file):
    # All time is in s.
    reader = csv.DictReader(input_file, delimiter = ',')
    result = list()
    for row in reader:
        if row['mrcuda_switch num_replay']:
            row['mrcuda_record time'] = float(row['mrcuda_record time'])
            row['mrcuda_switch time'] = float(row['mrcuda_switch time'])
            row['mrcuda_sync_mem time'] = float(row['mrcuda_sync_mem time'])
            row['mrcuda_replay time'] = row['mrcuda_switch time'] - row['mrcuda_sync_mem time']
            row['mrcuda_switch num_replay'] = int(row['mrcuda_switch num_replay'])
            result.append(row)
    return result

def plot_record_replay(input_data):
    properties = {
        'record_coef': 2.825 * (10 ** -7), # s
        'record_const': 0.3437 * (10 ** -3), # s
        'replay_coef': 1.031 * (10 ** -6), # s
        'replay_const': 1.2437, # s
    }

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    legend_list = list()

    x_values = [row['mrcuda_switch num_replay'] for row in input_data]

    p = ax1.scatter(
        x_values,
        [row['mrcuda_record time'] for row in input_data],
        c = COLOR[0],
        marker = 'o',
        s = 40
    )
    legend_list.append((p, 'Record Overhead (Measured)',))

    p = ax2.scatter(
        x_values,
        [row['mrcuda_replay time'] for row in input_data],
        c = COLOR[1],
        marker = 'o',
        s = 40
    )
    legend_list.append((p, 'Replay Overhead (Measured)',))

    x_values = sorted(set(x_values))

    ax1.plot(
        x_values, 
        [properties['record_coef'] * x + properties['record_const'] for x in x_values], 
        COLOR[0], 
        linewidth = 4
    )
    p = mlines.Line2D([], [], color = COLOR[0], linewidth = 4)
    legend_list.append((p, 'Record Overhead (Predicted)',))

    ax2.plot(
        x_values, 
        [properties['replay_coef'] * x + properties['replay_const'] for x in x_values], 
        COLOR[1], 
        linewidth = 4
    )
    p = mlines.Line2D([], [], color = COLOR[1], linewidth = 4)
    legend_list.append((p, 'Replay Overhead (Predicted)',))

    plt.legend(zip(*legend_list)[0], zip(*legend_list)[1],
        loc = 'lower right',
        prop = matplotlib.font_manager.FontProperties(size = 30, weight = 'bold')
    )
    #plt.xscale('log', basex = 2)
    #plt.yscale('log', basey = 10)
    ax1.set_xlim(xmin = 0)
    ax2.set_xlim(xmin = 0)
    ax1.set_ylim(ymin = 0)
    ax2.set_ylim(ymin = 0)

    ax1.set_xlabel('num_record (x10,000)', size = 30, weight = 'bold')
    ax1.set_ylabel('Record Time (ms)', size = 30, weight = 'bold')
    ax2.set_ylabel('Replay Time (s)', size = 30, weight = 'bold')

    ax1.set_xticklabels(['%d' % (int(label) / 10000,) for label in ax1.get_xticks().tolist()])

    for label in ax1.get_xticklabels():
        label.set_fontsize(25)
        label.set_fontweight('bold')

    ax1.set_yticklabels(['%d' % (float(label) * 1000,) for label in ax1.get_yticks().tolist()])

    for label in ax1.get_yticklabels():
        label.set_fontsize(25)
        label.set_fontweight('bold')
    for label in ax2.get_yticklabels():
        label.set_fontsize(25)
        label.set_fontweight('bold')

    plt.show()

def main():
    """
    Main function.
    """
    args = parseargs()

    if args.type == 'memsync':
        input_data = read_memsync_input(args.resultfile)
        plot_memsync(input_data)
    elif args.type == 'memsync-bw':
        input_data = read_memsync_input(args.resultfile)
        plot_memsync_bw(input_data)
    elif args.type == 'mhelper-nullker':
        input_data = read_mhelper_input(args.resultfile)
        plot_mhelper_nullker(input_data)
    elif args.type == 'mhelper-memcpybw':
        input_data = read_mhelper_input(args.resultfile)
        plot_mhelper_memcpybw(input_data)
    elif args.type == 'record-replay':
        input_data = read_record_replay_input(args.resultfile)
        plot_record_replay(input_data)

if __name__ == "__main__":
    main()

