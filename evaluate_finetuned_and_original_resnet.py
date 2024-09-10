import uuid
import random

from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights
import torch
import os
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset

class ResnetPGDAttacker:
    def __init__(self):
        '''
        The PGD attack on Resnet model.
        :param model: The resnet model on which we perform the attack
        :param dataloader: The dataloader loading the input data on which we perform the attack
        '''
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Nullify gradient for model params
        for p in self.model.parameters():
            p.requires_grad = False

    def pgd_attack(self, image, label, eps, alpha, steps):
        '''
        Create adversarial images for given batch of images and labels

        :param image: Batch of input images on which we perform the attack, size (BATCH_SIZE, 3, 224, 224)
        :param label: Batch of input labels on which we perform the attack, size (BATCH_SIZE)
        :return: Adversarial images for the given input images
        '''
        images = image.clone().detach().to(self.device)
        adv_images = images.clone()
        labels = label.clone().detach().to(self.device)

        # Starting at a uniformly random point within the eps ball
        random_noise = torch.zeros_like(adv_images).uniform_(-eps, eps)
        adv_images = adv_images + random_noise

        self.model.eval()
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


class EvaluationDatasetGenerator:
    def __init__(self, batch_size, batch_num, num_perturbations, save_path, no_perturbations):
        self.num_perturbations = num_perturbations
        self.no_perturbations = no_perturbations
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.save_path = save_path

        # Create the save directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)
        weights = ResNet50_Weights.DEFAULT
        self.resnet_transform = weights.transforms()  # PIL -> tensor
        self.pgd_attacker = ResnetPGDAttacker()

        self.ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, trust_remote_code=True)
        self.ds = self.ds.shuffle()
        self.ds = self.ds.filter(lambda example: example['image'].mode == 'RGB')
        self.ds = self.ds.take(self.batch_num * self.batch_size)
        self.ds = self.ds.map(self.preprocess_img)
        self.dataloader = DataLoader(self.ds, batch_size=self.batch_size)
        print(f"Evaluation Dataset Generator has been initialized. save path is {self.save_path}")

    def preprocess_img(self, example):
        example['image'] = self.resnet_transform(example['image'])
        return example

    def generate(self):
        print("Going to start generating")
        for i, batch in enumerate(tqdm(self.dataloader, total=self.batch_num)):
            if i == self.batch_num:
                break

            images, labels = batch["image"], batch["label"]

            if self.no_perturbations:
                self.save_images(images, labels)
                continue

            # img_batch = torch.stack(batch)
            # label_batch = torch.tensor([label] * len(img_batch))
            for _ in range(self.num_perturbations):
                # Generate random parameters for PGD attack
                random_eps = random.uniform(0.01, 0.3)
                random_alpha = random.uniform(0.01, 0.1)
                random_steps = random.randint(15, 20)

                # Perform the PGD attack
                perturbed_images = self.pgd_attacker.pgd_attack(images,
                                                                labels,
                                                                eps=random_eps,
                                                                alpha=random_alpha,
                                                                steps=random_steps)
                self.save_images(perturbed_images, labels)
            del images
            del labels

    def save_images(self, images, labels):
        for image, label in zip(images, labels):
            img_id = str(uuid.uuid4())
            save_file = os.path.join(self.save_path, f"class_{label}_img_{img_id}.pt")
            torch.save(image, save_file)

class Evaluation:
    def __init__(self, checkpoint_path, test_loader, evaluate_original=True, evaluate_finetuned=True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.evaluate_original = evaluate_original
        self.evaluate_finetuned = evaluate_finetuned
        self.checkpoint_path = checkpoint_path
        self.test_loader = test_loader

        if self.evaluate_original:
            self.original_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.original_model.to(self.device)
            self.original_model.eval()

        if self.evaluate_finetuned:
            self.fine_tuned_model = resnet50()
            self.load_model_from_checkpoint()
            self.fine_tuned_model.eval()

        print(f"evaluate original : {evaluate_original}")
        print(f"evaluate finetuned : {evaluate_finetuned}")

    def load_model_from_checkpoint(self):
        # Option 1: Use pytorch lightning
        # robustResnet = RobustResnet(train_loader, val_loader, test_loader)
        # finetuned_model = RobustResnet.load_from_checkpoint(checkpoint_path, train_loader=train_loader,
        #                                                     val_loader=val_loader, test_loader=test_loader)
        # print(f"Loaded model from {checkpoint_path}")

        # Option 2: Regular approach
        # Load the checkpoint
        checkpoint = torch.load(self.checkpoint_path)

        self.fine_tuned_model.load_state_dict(checkpoint['state_dict'])
        self.fine_tuned_model.to(self.device)
        self.fine_tuned_model.eval()

        # # Create a new instance of your model
        # finetuned_model = RobustResnet()  # Initialize with the required parameters
        #
        # # Load the model weights from the checkpoint
        # finetuned_model.load_state_dict(checkpoint['state_dict'])

    def evaluate_model(self):
        # Initialize accuracy metrics
        total = 0
        orig_correct = 0
        ft_correct = 0

        # Evaluate the original model
        for batch in tqdm(self.test_loader):
            images, labels = batch
            total += len(labels)

            images, labels = images.to(self.device), labels.to(self.device)

            # Original model predictions
            if self.evaluate_original:
                original_logits = self.original_model(images).softmax(1)
                original_predictions = original_logits.argmax(dim=1)
                orig_correct += torch.sum(original_predictions == labels).item()

            if self.evaluate_finetuned:
                ft_images = images.clone().detach().to(self.device)
                ft_labels = labels.clone().detach().to(self.device)

                # Fine-tuned model predictions
                fine_tuned_logits = self.fine_tuned_model(ft_images).softmax(1)
                fine_tuned_predictions = fine_tuned_logits.argmax(dim=1)
                ft_correct += torch.sum(fine_tuned_predictions == ft_labels).item()

        # Calculate accuracies
        result = {}
        if self.evaluate_original:
            original_accuracy = orig_correct / total
            print(f'Evaluation Original Model Accuracy: {original_accuracy * 100} %')
            result["original_accuracy"] = original_accuracy

        if self.evaluate_finetuned:
            fine_tuned_accuracy = ft_correct / total
            print(f'Evaluation Fine-Tuned Model Accuracy: {fine_tuned_accuracy * 100} %')
            result["fine_tuned_accuracy"] = fine_tuned_accuracy

        return result

class LocalDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = os.listdir(root_dir)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image from the file
        image_file = self.image_files[idx]
        image = torch.load(os.path.join(self.root_dir, image_file)).to(self.device)

        # Extract label from the filename
        label = int(os.path.basename(image_file).split("_")[1])
        label = torch.tensor(label).to(self.device)  # e.g., "class_0_img_3.pt" -> label = 0

        return image, label


def remove_prefix_from_state_dict(ckpt_path, new_ckpt_path, prefix='model.'):
    '''
    Pytorch lightning adds the model. as a prefix in the ckpt which has to be surgically removed
    '''

    # Load the checkpoint
    checkpoint = torch.load(ckpt_path)

    # Get the state_dict from the checkpoint
    state_dict = checkpoint['state_dict']

    # Create a new state_dict with updated keys
    new_state_dict = {key[len(prefix):] if key.startswith(prefix) else key: value
                      for key, value in state_dict.items()}

    # Replace the state_dict in the checkpoint with the new one
    checkpoint['state_dict'] = new_state_dict

    # Save the updated checkpoint to a new file
    torch.save(checkpoint, new_ckpt_path)

    print(f"Checkpoint saved to {new_ckpt_path}")


if __name__ == '__main__':
    # tensor = torch.zeros(3,224,224)
    # zero_tensor = torch.zeros(3,224,224)
    # assert not torch.equal(tensor, torch.zeros_like(tensor)), "Tensor is all zeros"
    BATCH_SIZE = 4
    BATCH_NUM = 250

    # creating local dataset for evaluation
    # eval_dataset_generator = EvaluationDatasetGenerator(batch_size=BATCH_SIZE,
    #                                                     batch_num=BATCH_NUM,
    #                                                     num_perturbations=2,
    #                                                     save_path="./perturbed",
    #                                                     no_perturbations=False)
    # eval_dataset_generator.generate()
    # #TODO. checking if the normal load_dataset is uniform generating!
    # exit(0)
    # evaluating the models
    checkpoint_normal_path = "./checkpoints/ft-checkpoint.ckpt"
    checkpoint_pgd_path = "./checkpoints/ft-pgd-checkpoint.ckpt"

    perturbed_data_path="./perturbed"
    clean_data_path="./clean"
    drive_path="./drive"
    new_data_path="./data_gen_test"

    # to_evaulate = [clean_data_path]
    to_evaulate = [new_data_path]
    # to_evaulate = [perturbed_data_path, clean_data_path, drive_path]
    for path in to_evaulate:
        print(f"path to evaluate is {path}")
        local_dataset = LocalDataset(root_dir=path)

        print(f"Evaluating model")
        local_ft_loader = DataLoader(local_dataset, batch_size=BATCH_SIZE)
        evaluation_ft = Evaluation(checkpoint_normal_path, local_ft_loader, evaluate_original=True, evaluate_finetuned=False)
        evaluation_ft.evaluate_model()

        # print(f"Evaluating the pgd model")
        # local_pgd_loader = DataLoader(local_dataset, batch_size=BATCH_SIZE)
        # evaluation_pgd = Evaluation(checkpoint_normal_path, local_pgd_loader, evaluate_original=True, evaluate_finetuned=False)
        # evaluation_pgd.evaluate_model()
        print("-" * 50)


'''
TODO. thoughts:
. Ensure dataloaders dont have any num workers or shuffle ( shuffle is actually okay but just to be safe )
. Generate clean data and check accuracy of original and new models X (getting high accruacy & poor in finetuned) 
    => have to retrain anyways
. On bad data check accuracy as well X => (poor accuracy for both models )
    => meaning current generation process locally is correct as the original model has bad accuracy, finetuned model is shit anyways
. Data in drive bad - check 2 zips locally => (data is not all zeros, getting very high accuracy in ft and avg in original) 
    => dataset creation was not done correctly!?
    => should be low in original model but is high instead
    => training has clearly confused the model someway 
. when generating the drive data, I didnt do model.eval() maybe thats a problem? -> try generating something locally that way 
    * gives bad accuracy correctly irrespective of the model.eval() in todo
. dont do any augmentations just generate like shit tons , is it evenly distributed? 
    -> maybe as long as we use only one dataset it is evenly distributed..
    -> perhaps better to just generate with one dataloader and zip things up.
    

. unzipping might be a problem with crucial folder being removed?
. Dataloader could be bad in colab notebook - check locally
. Regenerate


'''