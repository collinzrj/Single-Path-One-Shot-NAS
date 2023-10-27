def get_cifar_dataloaders():
        cifarset_train = torchvision.datasets.CIFAR10(root=os.path.join(args.data_root, args.dataset), train=False,
                                                download=True, transform=valid_transform)
        with open('./configs/cifar10_params.yaml', encoding='utf8') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            params = Params(**params)
        testset = AttackDataset(dataset=cifarset_train,
                                    synthesizer=PrimitiveSynthesizer(params, InputStats(cifarset_train)),
                                    percentage_or_count=0,
                                    random_seed=0,
                                    clean_subset=0)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                    shuffle=False, pin_memory=True, num_workers=8)
        attackset = AttackDataset(dataset=cifarset_train,
                                    synthesizer=PrimitiveSynthesizer(params, InputStats(cifarset_train)),
                                    percentage_or_count='ALL',
                                    random_seed=0,
                                    clean_subset=0,
                                    keep_label=True)
        attack_loader = torch.utils.data.DataLoader(attackset, batch_size=args.batch_size,
                                                    shuffle=False, pin_memory=True, num_workers=8)
        
        valset = AttackDataset(dataset=cifarset_val,
                                 synthesizer=synthesizer,
                                 percentage_or_count=POISON_PERCENTAGE,
                                 random_seed=0,
                                 clean_subset=0)
        noattack_set = AttackDataset(dataset=cifarset_val,
                                 synthesizer=synthesizer,
                                 percentage_or_count=0,
                                 random_seed=0,
                                 clean_subset=0)