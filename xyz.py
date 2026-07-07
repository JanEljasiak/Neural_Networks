import nbformat

path = "Backpropagation.ipynb"   # <-- podmień na właściwą ścieżkę do Twojego pliku

nb = nbformat.read(path, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code":
        if "execution_count" not in cell:
            cell["execution_count"] = None
        if "outputs" not in cell:
            cell["outputs"] = []

nbformat.write(nb, path)
print("Naprawiono i zapisano:", path)