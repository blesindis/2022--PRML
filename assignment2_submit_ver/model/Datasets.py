import torchtext.legacy.data as data


class MyDataset(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, src_data, trg_data, fields, **kwargs):        
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        examples = []
        for src_line, trg_line in zip(src_data, trg_data):
            src_line, trg_line = src_line.strip(), trg_line.strip()
            if src_line != '' and trg_line != '':
                examples.append(data.Example.fromlist([src_line, trg_line], fields))        

        super(MyDataset, self).__init__(examples, fields, **kwargs)