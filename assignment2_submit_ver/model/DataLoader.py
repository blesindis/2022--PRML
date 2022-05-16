class DataBundle():
    def __init__(self, paths):
        self.train = self._load_dataset(paths['train_en'], paths['train_zh'])
        self.dev = self._load_dataset(paths['dev_en'], paths['dev_zh'])
        self.test = self._load_dataset(paths['test_en'], paths['test_zh'])
        self.words_count = {}
        self.vocab = {}
        

    def _load_dataset(self, en_path, zh_path):
        en_set = []
        zh_set = []
        with open(en_path, 'r') as en_file:
            for line in en_file:
                en_set.append(line.strip())
        with open(zh_path, 'r') as zh_file:
            for line in zh_file:
                zh_set.append(line.strip())
        dataset = {'en': en_set, 'zh': zh_set}
        return dataset
    

    def process(self, min_freq=1):
        self._create_vocab('en')
        self._create_vocab('zh')
        self._trim_vocab(min_freq)

        
    def get_dataset(self, dataset_name):
        if dataset_name == 'train':
            return self.train
        elif dataset_name == 'dev':
            return self.dev
        elif dataset_name == 'test':
            return self.test
        else:
            raise ValueError('Invalid dataset name')
    

    def get_vocab(self, vocab_name):
        return self.vocab[vocab_name]


    def _create_vocab(self, name):
        words_count = {}
        for sentence in self.train[name]:
            for word in sentence.split():
                words_count[word] = words_count.get(word, 0) + 1
        self.words_count[name] = words_count

    
    def _trim_vocab(self, min_freq):
        self.vocab['en'] = ['<pad>', '<unk>']
        self.vocab['zh'] = ['<pad>', '<unk>']
        for word, count in self.words_count['en'].items():
            if count >= min_freq:
                self.vocab['en'].append(word)
        for word, count in self.words_count['zh'].items():
            if count >= min_freq:
                self.vocab['zh'].append(word)


    def print_info(self):
        print('Datasets:')
        print('\tTrain: {}'.format(len(self.train['en'])))
        print('\tDev: {}'.format(len(self.dev['en'])))
        print('\tTest: {}'.format(len(self.test['en'])))
        print('Vocabulary:')
        print('\tEn: {}'.format(len(self.vocab['en'])))
        print('\tZh: {}'.format(len(self.vocab['zh'])))

    
    def print_instances(self):
        for i in range(2):
            print("{}.\n{}\n{}".format(i+1, self.train['en'][i], self.train['zh'][i]))
        print('\n')
        print(self.vocab['en'][:10])
        print(self.vocab['zh'][:10])
        print('\n')
        

