#TODO da completare

def train(img_set_path, label_set_path, encoding_size, r, mode, lr, num_epochs, force_cpu=False):

    torch.cuda.empty_cache()
    device, batch_size = find_device_and_batch_size()


    transform = T.Compose([
                T.Resize((256, 512)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

    train_dst = Cityscapes(root='dataset',
                            split='train', transform=transform)
    val_dst = Cityscapes(root='dataset',
                         split='test', transform=transform) #use train?

    train_loader = data.DataLoader(
                train_dst, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
                val_dst, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)


    resnet = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=16)