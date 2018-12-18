import matplotlib.pyplot as plt


class ModelGraphs:

    @staticmethod
    def plot_acc(history, title="Model Accuracy"):
        """Imprime una gráfica mostrando la accuracy por epoch obtenida en un entrenamiento"""
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title(title)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

    @staticmethod
    def plot_loss(history, title="Model Loss"):
        """Imprime una gráfica mostrando la pérdida por epoch obtenida en un entrenamiento"""
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(title)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper right')
        plt.show()

    @staticmethod
    def plot_compare_losses(history1, history2, name1="Red 1",
                            name2="Red 2", title="Graph title"):
        """Compara losses de dos entrenamientos con nombres name1 y name2"""
        plt.plot(history1.history['loss'], color="green")
        plt.plot(history1.history['val_loss'], 'r--', color="green")
        plt.plot(history2.history['loss'], color="blue")
        plt.plot(history2.history['val_loss'], 'r--', color="blue")
        plt.title(title)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train ' + name1, 'Val ' + name1,
                    'Train ' + name2, 'Val ' + name2],
                   loc='upper right')
        plt.show()

    @staticmethod
    def plot_compare_accs(history1, history2, name1="Red 1",
                          name2="Red 2", title="Graph title"):
        """Compara accuracies de dos entrenamientos con nombres name1 y name2"""
        plt.plot(history1.history['acc'], color="green")
        plt.plot(history1.history['val_acc'], 'r--', color="green")
        plt.plot(history2.history['acc'], color="blue")
        plt.plot(history2.history['val_acc'], 'r--', color="blue")
        plt.title(title)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train ' + name1, 'Val ' + name1,
                    'Train ' + name2, 'Val ' + name2],
                   loc='lower right')
        plt.show()

    # grafica para dibujar multiples metricas de los modelos
    @staticmethod
    def plot_compare_multiple_metrics(history_array, names, colors, title="Graph title", metric='acc'):
        legend = []
        for i in range(0, len(history_array)):
            plt.plot(history_array[i].history[metric], color=colors[i])
            plt.plot(history_array[i].history['val_' + metric], 'r--', color=colors[i])
            legend.append('Train ' + names[i])
            legend.append('Val ' + names[i])

        plt.title(title)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(legend,
                   loc='lower right')
        plt.show()
