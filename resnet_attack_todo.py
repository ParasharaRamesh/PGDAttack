from torch.utils.data import DataLoader
import torch
from torch import nn
from tqdm import tqdm

import torch.nn.functional as F
class ResnetPGDAttacker:
    def __init__(self, model, dataloader: DataLoader):
        '''
        The PGD attack on Resnet model.
        :param model: The resnet model on which we perform the attack
        :param dataloader: The dataloader loading the input data on which we perform the attack
        '''
        self.model = model
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.adv_images = []
        self.labels = []
        self.eps = 0
        self.alpha = 0
        self.steps = 0
        self.acc = 0
        self.adv_acc = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Nullify gradient for model params
        for p in self.model.parameters():
            p.requires_grad = False

    def pgd_attack(self, image, label, eps=None, alpha=None, steps=None):
        '''
        Create adversarial images for given batch of images and labels

        :param image: Batch of input images on which we perform the attack, size (BATCH_SIZE, 3, 224, 224)
        :param label: Batch of input labels on which we perform the attack, size (BATCH_SIZE)
        :return: Adversarial images for the given input images
        '''
        if eps is None:
            eps = self.eps
        if alpha is None:
            alpha = self.alpha
        if steps is None:
            steps = self.steps

        images = image.clone().detach().to(self.device)
        adv_images = images.clone()
        labels = label.clone().detach().to(self.device)

        # Starting at a uniformly random point within the eps ball
        random_noise = torch.zeros_like(adv_images).uniform_(-eps, eps)
        adv_images = adv_images + random_noise

        for _ in range(steps):
            # Enable gradient tracking for adversarial images
            adv_images.requires_grad = True

            # Get model predictions and apply softmax
            outputs = self.model(adv_images).softmax(1)

            # Calculate loss
            loss = self.loss_fn(outputs, labels)

            # Compute gradient wrt images
            grad = torch.autograd.grad(
                loss, adv_images, retain_graph=False, create_graph=False
            )[0]
            adv_images = adv_images.detach()

            # Gradient update
            adv_images = adv_images + alpha * grad.sign()  # Update adversarial images using the sign of the gradient

            # Projection step
            # Clamping the adversarial images to ensure they are within the L∞ ball of eps radius of original image
            adv_images = torch.clamp(adv_images, images - eps, images + eps)

            adv_images = adv_images.detach()

        return adv_images  # Return the generated adversarial images

    def pgd_batch_attack(self, eps, alpha, steps, batch_num):
        '''
        Launch attack for many batches and save results as class features
        :param eps: Epsilon value in PGD attack
        :param alpha: Alpha value in PGD attack
        :param steps: Step value in PGD attack
        :param batch_num: Number of batches to run the attack on
        :return: Update attacker accuracy on original images, accuracy on adversarial images,
        and list of adversarial images
        '''
        self.model.eval()
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        adv_correct = 0
        correct = 0
        total = 0
        adv_images_lst = []
        for i, inputs in enumerate(tqdm(self.dataloader, total=batch_num)):
            if i == batch_num:
                break
            adv_images = self.pgd_attack(**inputs)
            with torch.no_grad():
                adv_outputs = self.model(adv_images).softmax(1)
                adv_predictions = adv_outputs.argmax(dim=1).cpu()
                outputs = self.model(inputs['image'].to(self.device)).softmax(1)
                predictions = outputs.argmax(dim=1).cpu()
            labels = inputs['label']
            adv_correct += torch.sum(adv_predictions == labels).item()
            correct += torch.sum(predictions == labels).item()
            total += len(labels)
            adv_images_lst.append(adv_images)
        self.adv_images = torch.cat(adv_images_lst).cpu()
        self.acc = correct / total
        self.adv_acc = adv_correct / total

    def compute_accuracy(self, batch_num):
        '''
        Compute model accuracy for specified number of data batches from self.dataloader
        :param batch_num: Number of batches on which we compute model accuracy
        :return: Update model accuracy
        '''
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, inputs in enumerate(tqdm(self.dataloader, total=batch_num)):
                if i == batch_num:
                    break
                inputs = {k: v.to(self.device) for (k, v) in inputs.items()}
                outputs = self.model(inputs['image']).softmax(1)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == inputs['label']).sum().item()
                total += predictions.size(0)
        self.acc = correct / total
