class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = config['Model']
        self.crit = config['Loss Function']
        self.opt = config['Optimizer']
        self.scheduler = config['Scheduler']


    def train(self, epochs):
        model = self.model
        crit = self.crit
        opt = self.opt
        scheduler = self.scheduler

        train_losses = []
        val_losses = []

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ion()
        fig.show()
        fig.canvas.draw()

        last_lr = float('inf')
        epoch_bar = tqdm(range(epochs))
        model.train()
        for epoch in epoch_bar:
            # Train
            loss_all = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data)
                label = data.y.to(device)
                loss = crit(output, label)
                loss.backward()
                loss_all += float(data.num_graphs * (loss.item()))
                optimizer.step()

            train_losses.append(loss_all / len(train_loader.dataset))

            # Validate
            with torch.no_grad():
                val_loss_all = 0
                for val_batch in val_loader:
                    val_data = val_batch.to(device)
                    val_label = val_data.y
                    out_val = model(val_data)
                    val_loss = crit(out_val, val_data.y)
                    val_loss_all += float(val_data.num_graphs * (val_loss.item()))
            val_losses.append(val_loss_all / len(val_loader.dataset))
            epoch_bar.set_description("Train: %.2e, val: %.2e" % (train_losses[-1], val_losses[-1]))

            if val_loss_all / len(val_loader.dataset) <= min(val_losses):
                # torch.save(model, self.save_loc)
                best_model = model

            ax.clear()
            plt.plot(train_losses, label="Training")
            plt.plot(val_losses, label="Validation")
            plt.yscale('log')
            fig.canvas.draw()
            plt.pause(0.05)

            if scheduler.get_lr()[0] != last_lr:
                last_lr = scheduler.get_lr()[0]
                print('Iter %i, Learning rate %f' % (epoch, last_lr))

            if scheduler:
                scheduler.step()

        return model, best_model, train_losses, val_losses
