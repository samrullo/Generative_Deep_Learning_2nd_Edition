# Replicating energy model

Replicating Energy model was quite easy. It was all about stacking convolutional layers with 
```torch.nn.Conv2d``` and specifying 

- number of input channels
- number of output channels
- kernel size
- stride
- padding (same in tensorflow usually means 1 in pytorch which will add padding of 1 pixels to all 4 sides of an image)

# Biggest challenges

## Generating samples with Langevin dynamics
First challenge was with replicating ```generate_samples``` function which uses Langevin dynamics to generate samples from a certain distribution.
The ideas is you have starting images whose pixel values are randomly drawen from uniform distribution.
Then you pass it through energy model and get energy score.
You compute gradients of this energy score against input images and updates images by adding gradients to it.
You also add a small amount of noise to the images.
This gives chance to explore various points in the distribution without falling to local minimum or maximum.

It turned out that PyTorch sets ```requires_grad``` attribute of inputs to ```False``` by default.
This is because usually you don't need to calculate gradients against ```inputs```.
Usually you are only interested in updating model parameters through gradient descent.

So to update input images through gradient descent multiple times, you need to explicitly set its ```requires_grad``` attribute to ```True```

But on top of that you have to detach input images from computation graph, because PyTorch will complain saying ```You cant set requires_grad of non-leaf variables```

This happens because input images are used as intermediate variables in computation graph of generating samples.

As a result input images that were ```leaf``` in the beginning, become ```non-leaf``` after one pass.

## Training
When training the model was not converging at all.
I couldn't find the reason for a long time.
Then the climax turned out to be with ```generate_samples``` again.
Tensorflow was passing every image in the batch through energy model and computing energy scores.
Then it would compute gradients of against every single image using corresponding energy score of every image.
In PyTorch I had to apply ```backward``` to compute gradients.
But ```backward``` method can only be applied to scalar values.
So initially I was computing the ```mean``` of energy scores of a batch and applying backward on the resulting mean value.
That turned out to be the problem.
Instead of mean I had to use ```sum```, as in ```outscore = torch.sum(outscores, dim=0)```