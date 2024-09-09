import torch.nn as nn
import torch


class MinibatchCKA(nn.Module):
    expansion = 1

    def __init__(self, num_layers, device,
                 num_layers2=None, 
                 across_models=False):
        super(MinibatchCKA, self).__init__()

        if num_layers2 is None:
            num_layers2 = num_layers
        
        self.hsic_accumulator = torch.zeros(num_layers, num_layers)
        self.hsic_accumulator = self.hsic_accumulator.to(device)
        self.across_models = across_models
        if across_models:
            self.hsic_accumulator_model1 = torch.zeros(num_layers,).to(device)
            self.hsic_accumulator_model2 = torch.zeros(num_layers2,).to(device)

            
            #print(self.hsic_accumulator_model1.shape,
            #    self.hsic_accumulator_model2.shape )
            
        print(self.hsic_accumulator.shape)
        print(device)


    def _generate_gram_matrix(self, x):
        """Generate Gram matrix and preprocess to compute unbiased HSIC.
        https://inst.eecs.berkeley.edu/~ee127/sp21/livebook/def_Gram_matrix.html
        This formulation of the U-statistic is from Szekely, G. J., & Rizzo, M.
        L. (2014). Partial distance correlation with methods for dissimilarities.
        The Annals of Statistics, 42(6), 2382-2412.
        """
        #print("_generate_gram_matrix")
        x = torch.reshape(x, (x.shape[0], -1))
        gram = torch.matmul(x, x.T)
        n = gram.shape[0]
        gram.fill_diagonal_(0)
        means = torch.sum(gram, axis=0) / (n - 2)
        means = means - ( torch.sum(means) /  (2 * (n-1))   )

        gram -= means[:, None]
        gram -= means[None, :]
        gram.fill_diagonal_(0)
        gram = torch.reshape(gram, (-1,))

        # print(gram.shape)
        return gram

    def update_state(self, activations):
        """
        Accumulate minibatch HSIC values.

        Args:
            activations: a list of activations for all layers
        """

        #for x in activations:
        #    grams = self._generate_gram_matrix(x)
        #    print(grams.shape)
        # for idx, x in enumerate(activations):
        #     if idx == 5:
        #         print("activations", x.sum())

        print(f'activation length: {len(activations)}')
        layer_grams = [self._generate_gram_matrix(x) for x in activations]
        layer_grams = torch.stack(layer_grams, dim=0)
        print(layer_grams.shape)
        hsic = torch.matmul(layer_grams, layer_grams.T)
    
        self.hsic_accumulator += hsic
        #print(layer_grams[5].shape, layer_grams[5])
        #import sys
        #sys.exit(0)
        #print(hsic.detach().cpu().numpy()[5])

    
    def result(self):
        mean_hsic = self.hsic_accumulator
        #print(mean_hsic.detach().cpu().numpy()[5])
        if self.across_models:
            raise NotImplementedError
        else:
            normalization = torch.sqrt(torch.diag(mean_hsic, 0))
            mean_hsic /= normalization[:, None]
            mean_hsic /= normalization[None, :]
        #print(mean_hsic.detach().cpu().numpy()[19])
        return mean_hsic
