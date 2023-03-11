import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaGradCAM:
    """
    Description
    -----------
        Vanilla GradCAM class.

    Arguments
    -----------
        model (nn.Module): Model
        target_layer (int): Target layer

    Returns
    -----------
        None
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

    @staticmethod
    def get_index(dictionay: dict, query: str):
        """
        Description:
        -----------
            Returns the index of the key in the dictionary.

        Arguments:
        -----------
            dictionay (dict): Dictionary
            query (str): Key

        Returns:
        -----------
            index (int): Index of the key
        """
        for key, value in dictionay.items():
            if value == query:
                return key

    def save_activations(self, module, input, output):
        """
        Description:
        -----------
            Saves the output of the convolution layer.

        Arguments:
        -----------
            module (nn.Module): Module
            input (tuple): Input
            output (tuple): Output

        Returns:
        -----------
            None
        """
        self.activations = output.detach()

    def save_grad_output(self, module, grad_input, grad_output):
        """
        Description:
        -----------
            Saves the gradient of the output with respect to the weights.

        Arguments:
        -----------
            module (nn.Module): Module
            grad_input (tuple): Gradients of the input
            grad_output (tuple): Gradients of the output

        Returns:
        -----------
            None
        """
        self.gradients = grad_output[0].detach()

    def get_conv_layers(self):
        """
        Description:
        -----------
            Returns a dict of all the convolutional layers in the model.

        Arguments:
        -----------
            self (VanillaGradCAM): VanillaGradCAM object

        Returns:
        -----------
            conv_layers (dict): Dict of all the convolutional layers in the model.
        """

        # we will save the conv layer weights in this dict
        model_weights = dict()
        # we will save the  conv layers in this dict
        conv_layers = dict()
        # get all the model children as dict
        model_children = dict(self.model.named_children())
        # counter to keep count of the conv layers
        counter = 0
        # append all the conv layers and their respective wights to the list
        for name, child in model_children.items():
            if isinstance(child, nn.Conv2d):
                counter += 1
                model_weights[name] = child.weight
                conv_layers[name] = child
            elif isinstance(child, nn.Sequential):
                for j, _ in enumerate(child):
                    for sub_name, sub_child in child[j].named_children():
                        if isinstance(sub_child, nn.Conv2d):
                            counter += 1
                            model_weights[
                                name + "_" + str(j) + "_" + sub_name
                            ] = sub_child.weight
                            conv_layers[
                                name + "_" + str(j) + "_" + sub_name
                            ] = sub_child
            elif isinstance(child, nn.Linear):
                model_weights[name] = child.weight
                conv_layers[name] = child

        return conv_layers

    def hook_on_target_layer(self, conv_layers_dict: dict, backward=False):
        """
        Does a forward pass on convolutions, hooks the function at given layer
        """
        for i, (name, module) in enumerate(conv_layers_dict.items()):
            if int(i) == self.target_layer:
                if backward:
                    backward_hook_handle = module.register_full_backward_hook(
                        self.save_grad_output
                    )
                    return backward_hook_handle
                else:
                    forward_hook_handle = module.register_forward_hook(
                        self.save_activations
                    )
                    return forward_hook_handle

    def generate_class_activation_maps(
        self, input_tensor, classes_dict: dict, target_class: str = None
    ):
        """
        Description:
        -----------
            Generates the class activation maps for the input image.

        Arguments:
        -----------
            input_image (torch.Tensor): Input image
            classes_dict (dict): Dictionary of classes
            target_class (str): Target class

        Returns:
        -----------
            cam (torch.Tensor): Class activation map
        """
        # Query all the potential `rectified convolutional feature maps` of interest
        conv_layers = self.get_conv_layers()

        # Hook the function at the specified layer
        forward_hook_handle = self.hook_on_target_layer(
            conv_layers_dict=conv_layers, backward=False
        )
        backward_hook_handle = self.hook_on_target_layer(
            conv_layers_dict=conv_layers, backward=True
        )

        # Forward pass
        raw_scores = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(raw_scores)
        else:
            target_class = self.get_index(dictionay=classes_dict, query=target_class)

        # Target for backprop
        one_hot_output = torch.zeros_like(raw_scores)
        one_hot_output[0][target_class] = 1

        # Zero grads
        self.model.zero_grad()

        # =============================== Backward pass with specified target: =============================== #
        # --> backward pass is performed on the raw scores (output from the last fully connected layer).
        # The gradient is set to a one-hot encoded tensor representing the target class.
        # retain_graph=True is used to keep the computation graph in memory for further usage.
        raw_scores.backward(gradient=one_hot_output, retain_graph=True)

        # =============================== Get hooked gradients: =============================== #
        # --> mean of the gradients with respect to the spatial dimensions (height and width) is computed.
        # It represents the importance of each feature map in the target layer with respect to the target class.
        weights = torch.mean(self.gradients, dim=(2, 3))

        # =============================== Dot product between the weights and the activations =============================== #
        # Activations are reshaped to a 2D tensor for matrix multiplication.
        # This gives us the Class Activation Map (CAM) for the target layer.
        cam = torch.matmul(weights, self.activations.view(self.activations.size(1), -1))

        # =============================== Reshape and normalize the CAM =============================== #
        # CAM is reshaped to its original spatial dimensions (height and width) and the number of channels is set to 1.
        cam = cam.view(cam.size(0), 1, *self.activations.shape[2:])

        # =============================== ReLU activation function =============================== #
        # Applied to the CAM to ensure that only positive activations are used.
        cam = F.relu(cam)

        # =============================== Up-sample the CAM to the original input image size =============================== #
        # The CAM is up-sampled to the original input image size using bilinear interpolation.
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # =============================== Normalize the CAM =============================== #
        # The CAM is normalized to the range [0, 1] using min-max normalization.
        cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam))

        # =============================== Squeeze the CAM =============================== #
        # The CAM is squeezed to remove the channel dimension.
        cam = cam.squeeze(1)

        # Remove hooks
        forward_hook_handle.remove()
        backward_hook_handle.remove()
        return cam

    @staticmethod
    def display_heatmap(
        img: np.ndarray,
        cam: np.ndarray,
        use_rgb: bool = False,
        colormap: int = cv2.COLORMAP_JET,
        image_weight: float = 0.5,
    ) -> np.ndarray:
        """
        Description:
        -----------
            Displays the class activation map over the input image.

        Arguments:
        -----------
            img (np.ndarray): Original input image
            cam (np.ndarray): Class activation map mask
            use_rgb (bool): Whether to use RGB or BGR
            colormap (int): OpenCV color map
            image_weight (float): Weight of the image

        Returns:
        -----------
            np.ndarray: Original input mage with heatmap overlay.
        """
        # Convert Class Activation Maps to numpy
        cam = cam.permute(1, 2, 0).detach().numpy()

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        heatmap = np.float32(heatmap) / 255

        if np.max(img) > 1:
            raise ValueError(
                "Bad input shape: {type(img)}; input image should be of type np.float32 in the range [0, 1]"
            )

        if image_weight < 0 or image_weight > 1:
            raise ValueError(
                f"image_weight should be in the range 0 <= x <= 1 but got {image_weight}"
            )

        cam = (1 - image_weight) * heatmap + image_weight * img
        cam = cam / np.max(cam)

        return cam
