
def padding_dataset(dataset, batch_size):
    sample_num = len(dataset)
    if sample_num % batch_size == 0:
        steps = len(dataset) // batch_size
    else:
        steps = len(dataset) // batch_size + 1
    padding = steps * batch_size - sample_num
    dataset += dataset[:padding]
    return dataset, sample_num, steps
