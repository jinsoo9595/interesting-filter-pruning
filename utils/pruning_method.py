import numpy as np
from kerassurgeon import Surgeon
from utils.geometric_method import geometric_median

def pruning_method_conv(model, layer_to_prune, pruning_amount, method):

    if method == 'L1norm':
        # Load surgeon package
        surgeon = Surgeon(model)

        # Store weights from conv layers [0][1] = [weight][bias] Hence, [0] to store weights only
        conv_layer_weights = [model.layers[i].get_weights()[0] for i in layer_to_prune]
        print('number of layer to prune: ' + str(len(conv_layer_weights)))

        for i in range(len(conv_layer_weights)):
            if pruning_amount[i] == 0:
                continue
            weight = conv_layer_weights[i]
            num_filters = len(weight[0, 0, 0, :])
            print('total number of filter: ' + str(num_filters))
            weight_removable = {}

            # compute L1-nom of each filter weight and store it in a dictionary(weight_removable)
            for j in range(num_filters):
                L1_norm = np.sum(abs(weight[:, :, :, j]))
                filter_number = 'filter_{}'.format(j)
                weight_removable[filter_number] = L1_norm

            # sort the filter according to the ascending L1 value
            weight_removable_sort = sorted(weight_removable.items(), key=lambda kv: kv[1])

            # 'filter_24'(string) -> 24(int),
            # extracting filter number from '(filter_2, 0.515..), eg) extracting '2' from '(filter_2, 0.515..)
            remove_channel = [int(weight_removable_sort[i][0].split("_")[1]) for i in range(0, pruning_amount[i])]

            # delete filters with lowest scores
            surgeon.add_job('delete_channels', model.layers[layer_to_prune[i]], channels=remove_channel)

        model_pruned = surgeon.operate()

        return model_pruned

    elif method == 'geometric_median_conv':
        # Load surgeon package
        surgeon = Surgeon(model)

        # Store weights from conv layers [0][1] = [weight][bias] Hence, [0] to store weights only
        conv_layer_weights = [model.layers[i].get_weights()[0] for i in layer_to_prune]
        print('number of layer to prune: ' + str(len(conv_layer_weights)))

        for i in range(len(conv_layer_weights)):
            if pruning_amount[i] == 0:
                continue
            weight = conv_layer_weights[i]
            num_filters = len(weight[0, 0, 0, :])
            print('total number of filter: ' + str(num_filters))
            weight_removable = {}
            # 1. Reduce dimension 4D -> 2D
            # 2. Normalization (L1, L2)
            # 3. Sort norm value => get index order
            # 4. Sort weight by index order
            # 5. Calculate distance between coordinates
            # 6. Sum distance (of each filter)
            norm_val = geometric_median(weight, "euclidean", "L1")

            # compute L1-nom of each filter weight and store it in a dictionary(weight_removable)
            for j in range(num_filters):
                filter_number = 'filter_{}'.format(j)
                weight_removable[filter_number] = norm_val[j]

            # sort the filter according to the ascending L1 value
            weight_removable_sort = sorted(weight_removable.items(), key=lambda kv: kv[1])
            # print(weight_removable_sort)

            # 'filter_24'(string) -> 24(int),
            # extracting filter number from '(filter_2, 0.515..), eg) extracting '2' from '(filter_2, 0.515..)
            remove_channel = [int(weight_removable_sort[i][0].split("_")[1]) for i in range(0, pruning_amount[i])]
            # print(remove_channel)

            # delete filters with lowest scores
            surgeon.add_job('delete_channels', model.layers[layer_to_prune[i]], channels=remove_channel)

        model_pruned = surgeon.operate()

        return model_pruned


