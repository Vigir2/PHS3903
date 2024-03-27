import tkinter as tk

def run_program():
    # Récupérer les valeurs des champs
    parametre1 = entry_parametre1.get()
    parametre2 = entry_parametre2.get()
    # Exécuter le programme avec les paramètres fournis
    # Ici, vous pouvez appeler votre fonction ou votre script principal avec les paramètres fournis
    print("Paramètre 1:", parametre1)
    print("Paramètre 2:", parametre2)
    # Fermer la fenêtre
    root.destroy()

# Créer la fenêtre principale
root = tk.Tk()
root.title("Interface Utilisateur")

# Créer des libellés et des champs de saisie pour les paramètres d'entrée
label_parametre1 = tk.Label(root, text="Paramètre 1:")
label_parametre1.grid(row=0, column=0, padx=10, pady=5)
entry_parametre1 = tk.Entry(root)
entry_parametre1.grid(row=0, column=1, padx=10, pady=5)

label_parametre2 = tk.Label(root, text="Paramètre 2:")
label_parametre2.grid(row=1, column=0, padx=10, pady=5)
entry_parametre2 = tk.Entry(root)
entry_parametre2.grid(row=1, column=1, padx=10, pady=5)

# Créer un bouton pour exécuter le programme et fermer la fenêtre
run_button = tk.Button(root, text="Exécuter", command=run_program)
run_button.grid(row=2, columnspan=2, padx=10, pady=10)

# Lancer la boucle principale de l'interface utilisateur
root.mainloop()
