# Question 1
I am going to use loss_fn = losses.SparseCategoricalCrossentropy() to train LSTMMOdel we defined above together. I want to understand how this loss function exactly works. I know that my model will output tensor of shape (batch_size, vocab_size) where each record represents a list of probabilities for each token in vocabulary. Then I have target or y with shape (batch_size, number_of_tokens_in_a_record). How loss function is going to compare these two?

**Answer**
The `SparseCategoricalCrossentropy` loss function is used when you have a classification problem with multiple classes, and each example belongs to exactly one class. In the context of training an LSTM for a task like text generation or language modeling, your model's output is indeed a tensor of shape `(batch_size, sequence_length, vocab_size)`, where each entry along the `vocab_size` dimension is the predicted probability of the corresponding word in the vocabulary.

The target tensor `y`, in this case, would typically be of shape `(batch_size, sequence_length)`, and it contains the integer indices of the correct words in the vocabulary. Each entry in `y` is an integer corresponding to the index of the correct word in the vocabulary.

Here’s how `SparseCategoricalCrossentropy` works in this scenario:

1. **Logits to Probabilities**: If you don't apply a softmax activation function to the last layer of your model, you need to set `from_logits=True` in `SparseCategoricalCrossentropy`. This ensures that the function applies a softmax function to convert the output logits to probabilities.

2. **Selecting the Relevant Probabilities**: For each example in the batch, the loss function looks at the position in the sequence corresponding to the true class in `y` and selects the predicted probability of that class from the model’s output. For example, if `y` for a particular example is `[2]`, the loss function would look at the predicted probabilities for class `2`.

3. **Calculating Log Loss**: The loss function then calculates the negative log likelihood of the true class. The formula for this is `-log(p)`, where `p` is the predicted probability of the true class. This value is low (i.e., a low loss) when the predicted probability of the true class is high, and it is high (i.e., a high loss) when the predicted probability of the true class is low.

4. **Averaging over the Batch**: Finally, the loss function averages the loss over all examples in the batch to produce the final loss value.

This process effectively penalizes predictions that are confident but wrong and rewards predictions that are confident and correct, helping the model to learn the correct classifications over time.

# Question 2
why is it called Sparse

**Answer**
The term "Sparse" in `SparseCategoricalCrossentropy` refers to how the target classes are represented.

In a classification problem, you can represent the targets in two main ways:

1. **One-hot Encoding**: In this representation, each target class is represented as a vector of zeros with a one in the position corresponding to the class label. For example, for a problem with three classes, an instance of class 1 would be represented as `[1, 0, 0]`, class 2 as `[0, 1, 0]`, and class 3 as `[0, 0, 1]`. 

2. **Sparse Encoding**: In this representation, each target class is represented as a single integer. Using the same example, an instance of class 1 would be represented as `0`, class 2 as `1`, and class 3 as `2`.

`SparseCategoricalCrossentropy` is designed to be used with target classes in sparse encoding. It expects the targets to be provided as integers rather than one-hot encoded arrays. This is more memory-efficient, especially when dealing with a large number of classes, as is typically the case in language modeling where the vocabulary size can be quite large.

In contrast, if you were using one-hot encoding for your targets, you would use the `CategoricalCrossentropy` loss function. Both loss functions perform the same computation of categorical cross-entropy loss, but they expect the target classes to be formatted differently.

So in summary, it is called "Sparse" because it is designed to work with target class labels that are represented in a sparse format (as integers) rather than a dense one-hot encoded format.