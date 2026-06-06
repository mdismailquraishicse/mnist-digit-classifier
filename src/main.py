from train import Trainer


epochs = 5
trainer = Trainer(learning_rate=0.001)

def main():

    print("Hello from mnist-digit-classifier!")
    trainer.train(epochs=epochs)
    trainer.evaluate()


if __name__ == "__main__":
    main()
