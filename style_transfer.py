from __future__ import print_function
from datetime import datetime
import numpy as np

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

# import kornia

class InputImages:
    """
    Class used to load and make the necessary manipulations to the images
    used as an input to the network.
    """
    def __init__(self):
        self.image_size = 512 if torch.cuda.is_available() else 128
        self.loader = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
        self.unloader = transforms.ToPILImage()
        self.style_image = None
        self.content_image = None
        self.style_image_path = './data/images/picasso.jpg'
        self.content_image_path = './data/images/dancing.jpg'

    def resize_to_square(self, image):
        """
        Resizes the input image to a square image by reducing the
        larger side of the image to the smaller one.
        """
        width, height = image.size
        if width == height:
            return image
        else:
            if width > height:
                return image.crop((0, 0, height, height))
            else:
                return image.crop((0, 0, width, width))


    def image_loader(self, image_name, device=torch.device('cpu')):
        """
        Loads the image from the image_name path as a tensor with values
        normalized between 0 and 1
        """
        image = Image.open(image_name)
        image = self.resize_to_square(image)
        image = self.loader(image).unsqueeze(0)
        print('Tensor Image: ', image.size())
        return image.to(device, torch.float)

    def get_predefined_style_image(self, device=torch.device('cpu')):
        """Loads the predefined image o a picasso painting as the style image"""
        return self.image_loader(self.style_image_path, device)

    def get_predefined_content_image(self, device=torch.device('cpu')):
        """Loads the predefined image o a dancer as the content image"""
        return self.image_loader(self.content_image_path, device)

    def get_predefined_images(self, device=torch.device('cpu')):
        """Loads the predefined images."""
        self.style_image = self.get_predefined_style_image(device)
        self.content_image = self.get_predefined_content_image(device)
        print("Predefined images are now set.")
        return self.style_image, self.content_image

    def get_images(self, style_image_path, content_image_path, device=torch.device('cpu')):
        """
        Loads the images if any paths are passed as parameters otherwise it loads
        the predefined.
        """
        if None not in (style_image_path, content_image_path):
            self.style_image = self.image_loader(style_image_path, device)
            self.content_image = self.image_loader(content_image_path, device)
            return self.style_image, self.content_image
        if all(par is None for par in [style_image_path, content_image_path]):
            return self.get_predefined_images(device)

        self.style_image = (
            self.get_predefined_style_image(device)
            if (style_image_path is None)
            else self.image_loader(style_image_path, device)
        )
        self.content_image = (
            self.get_predefined_content_image(device)
            if content_image_path is None
            else self.image_loader(content_image_path, device)
        )
        return self.style_image, self.content_image

    def verify_images(self, style_image, content_image):
        """Checks if the two images are of the same size."""
        assert style_image.size() == content_image.size(), \
            "The two images have different sizes. They \
             must match in size in order to continue."

    def imshow(self, tensor_image, title=None, is_save=False):
        """Plots the image given as a tensor."""
        image = tensor_image.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)      # remove the fake batch dimension
        image = self.unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        if is_save:
            now = datetime.now()
            image.save('output_{}'.format(datetime.timestamp(now)), 'JPEG')
        plt.pause(1) # pause a bit so that plots are updated

    def show_images(self):
        """Plots the images if they were set"""
        plt.ion()
        plt.figure()
        if self.style_image is not None:
            self.imshow(self.style_image, 'Style Image')
        else:
            print('The style image has not been set')
        plt.figure()
        if self.content_image is not None:
            self.imshow(self.content_image, 'Content Image')
        else:
            print('The content image has not been set')

class ContentLoss(nn.Module):
    """
    Class used to measure the content distance between the feature map of
    layer L of the input image X and the feature map of layer L of the content
    image C given by w_{CL} * D^{L}_{C}(X, C) where
    D^{L}_{C}(X, C) = || F_{XL} - F_{CL} ||^2 (mean squared error). It receives
    F_{CL} as the constructors parameter.
    TODO: Convert it into a proper loss function implemented through an autograd
    function to recompute the gradient in the backward method.
    """
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()
        self.loss = None

    def forward(self, layer_input):
        """
        It computes the content loss between the input which the F_{XL}
        and the target which is defined in the constructor and is F_{CL}
        """
        # self.loss = F.mse_loss(layer_input, self.target)
        self.loss = F.l1_loss(layer_input, self.target)
        return layer_input

class StyleLoss(nn.Module):
    """
    Class used to compute the mean squared error of the gram matrix
    of the feature maps of the input X at layer L, G_{XL}, and the
    gram matrix of the feature maps of the style image S at layer L:
    || G_{XL} - G_{SL} ||^2
    """

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.gram_product_style = self.gram_matrix(target_feature).detach()
        self.loss = None

    def forward(self, layer_input):
        """
        It computes the style loss between the input which the G_{XL}
        and the target which is defined in the constructor and is G_{SL}
        """
        gram_product_input = self.gram_matrix(layer_input)
        # self.loss = F.mse_loss(gram_product_input, self.gram_product_style)
        self.loss = F.l1_loss(gram_product_input, self.gram_product_style)
        return layer_input

    def gram_matrix(self, layer_input):
        """
        Function to compute the gram matrix which is the result of a given
        matrix by its own transposed. In this particular case it is a
        reshaped version version of the feature maps F_{XL} reshaped so
        that to form a K*N matrix where K is the number of feature maps
        at layer L and N is the length of any vectorized feature map F^{k}_{XL}.
        It must be normalized by dividing each element by the total number of
        elements in the matrix.
        batch_size = 1
        feature_channels -> number of features maps K = a*b
        (c, d) -> dimensions of a features map. N = c*d
        """
        (batch_size, feature_channels,
         feature_map_height, feature_map_width) = layer_input.size()

        features_channels = batch_size * feature_channels
        feature_map_size = feature_map_height * feature_map_width
        features = layer_input.view(features_channels, feature_map_size)
        gram_product = torch.mm(features, features.t())
        total_elements = features_channels * feature_map_size
        return gram_product.div(total_elements)

class VggNet:
    """
    Class used to initialiaze a vgg19 pretrained network using the features
    sequential module since the output of each convolutional layer is needed
    for the purpose of this network.
    """
    def __init__(self, device):
        self.cnn = models.vgg19(pretrained=True).features.to(device).eval()
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    def get_normalization_mean(self):
        """Returns the normalization mean tensor"""
        return self.cnn_normalization_mean

    def get_normalization_std(self):
        """Returns the normalization standard deviation tensor"""
        return self.cnn_normalization_std

    def get_cnn(self):
        """Returns the the VGG 19 Convolutional Network"""
        return self.cnn

    def visualize_feature_maps(self, input, layer_number, feature_map_number):
        self.visualizations = []
        self.inputs = []

        def hook_fn(module, layer_input, layer_output):
            # print('module: ', module)
            # print('input', input)
            self.visualizations.append(layer_output)
            self.inputs.append(layer_input)
        
        for name, layer in self.cnn._modules.items():
            if isinstance(layer, nn.ReLU):
                layer.register_forward_hook(hook_fn)
        
        output = self.cnn(input)
        # print(self.visualizations.keys())
        plt.ioff()
        for layer_index, layer in enumerate(self.visualizations):
            if layer_index == layer_number:
                try:
                    feature_map = np.squeeze(
                        layer[0, feature_map_number]
                    ).data.numpy()
                    feature_map_input = np.squeeze(
                        self.inputs[layer_index][0][0, feature_map_number]
                    ).data.numpy()
                except:
                    print('Something went wrong.');
                    break
            # print(self.visualizations[layer_name].size())
                plt.ioff()
                plt.figure()
                plt.imshow(feature_map, cmap='gray');
                plt.figure()
                plt.imshow(feature_map_input, cmap='gray')
                break
            # print(self.visualizations[layer_name][0][0].size())
        plt.show()


class Normalization(nn.Module):
    """
    Class used to create a module to normalize the input images so
    that it can be easily be put into a Sequential module(nn.Sequential)
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, image):
        """Normalizes the image."""
        return (image - self.mean) / self.std


class StyleTransferNetwork:
    """
    Class used to define the neural network for the style transfer
    algorithm described by Gatys in https://arxiv.org/pdf/1508.06576.pdf
    It basically computes style and content losses at different layers of
    a given CNN in order to modify the input image according to them.
    It is characterized by the
        device -> where everything will be computed,
        normalization mean/standard -> are applied to the images used in
        deviation                      order to be correctly read by the
                                       original Neural Network.
        style_img -> The image from which the style is extracted in order
                     to be applied to the processed image
        content_img -> The image to which the style of the style_img will
                       be applied. It is used both as an input image and
                       as the content image used to exatract the needed
                       feature maps.
    """
    def __init__(self, cnn, normalization_mean, normalization_std,
                 style_img, content_img, use_random_noise=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.content_layers_default = ['relu_3_2']
        # self.style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.style_layers_default = ['relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1']
        self.cnn = cnn
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.style_image = style_img
        self.content_image = content_img
        self.input_image = (
            torch.randn(content_img.data.size(), device=self.device)
            if use_random_noise
            else content_img.clone().to(self.device)
        )

    def get_style_model_and_losses(self):
        """
        This function is used to redesign the original VGG 19 network
        by adding the style loss and the content loss layers where needed.
        Basically it adds style loss or content loss computations after
        predefined networks.
        """
        cnn = copy.deepcopy(self.cnn)

        # normalization module
        normalization = Normalization(self.normalization_mean,
                                      self.normalization_std).to(self.device)

        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        block, number = 1, 1  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                name = 'conv_{}_{}'.format(block, number)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}_{}'.format(block, number)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
                number += 1
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(block)
                layer = nn.AvgPool2d(layer.kernel_size, layer.stride)
                block += 1
                number = 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(block)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            
            print(name)
            model.add_module(name, layer)

            if name in self.content_layers_default:
                # add content loss:
                target = model(self.content_image).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(block), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers_default:
                # add style loss:
                target_feature = model(self.style_image).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(block), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def get_input_optimizer(self):
        """
        Defines the method used for the gradient descent which is  as suggested by
        Leon Gatys himself
        """
        # this line to show that input is a parameter that requires a gradient
        # optimizer = optim.LBFGS([self.input_image.requires_grad_()])
        optimizer = optim.Adam([self.input_image.requires_grad_()], lr=0.5)
        return optimizer

    def run_style_transfer(self, num_steps=1024, style_weight=1000.,
                           content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses()
        optimizer = self.get_input_optimizer()

        print('Optimizing..')
        run = [0]
        image_batch, image_channels, image_height, image_width = self.content_image.data.size()
        transform = nn.Sequential(
            kornia.augmentation.RandomResizedCrop(
                size=(image_height, image_width), scale=(.97, 1.), ratio=(.97, 1.03)),
            kornia.augmentation.RandomRotation(degrees=1.))
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image
                self.input_image.data.clamp_(0, 1)

                optimizer.zero_grad()
                self.input_image = transform(self.input_image)
                image_channels, image_width, image_height = self.input_image.size()
                self.input_image = self.input_image.view(
                    -1, image_channels, image_width, image_height).clamp_(0, 1)
                model(self.input_image)
                style_score = 0
                content_score = 0

                for style_loss in style_losses:
                    style_score += style_loss.loss
                for class_loss in content_losses:
                    content_score += class_loss.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        self.input_image.data.clamp_(0, 1)

        return self.input_image


def main():
    """Function used to execute the script"""
    style_transfer_device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )
    input_images = InputImages()
    style_image, content_image = input_images.get_images(
        './data/images/picasso.jpg',
        None,
        device=style_transfer_device
    )
    input_images.verify_images(style_image, content_image)
    input_images.show_images()
    vgg19 = VggNet(style_transfer_device)
    # activations = vgg19.visualize_feature_maps(style_image, 0, 8)
    style_transfer = StyleTransferNetwork(
        vgg19.get_cnn(),
        vgg19.get_normalization_mean(),
        vgg19.get_normalization_std(),
        style_image,
        content_image,
        True
    )
    output = style_transfer.run_style_transfer(num_steps=1024, style_weight=10000)

    plt.figure()
    input_images.imshow(output, title='Output Image', is_save=True)

    # sphinx_gallery_thumbnail_number = 4
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
