import os
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

# 创建保存路径
save_path = r'G:\2-thombin data\1-software\4VS\4VS_Cluster_100\100_conpunds'
os.makedirs(save_path, exist_ok=True)

# 从每个簇中选择XScore最低的分子，并显示分子名称
selected_compounds = []
selected_compounds_names = []

for i in range(100):
    if ward_library[i]:  # 确保簇不为空
        min_xscore = float('inf')
        min_xscore_mol = None
        min_xscore_mol_name = None
        
        for mol in ward_library[i]:
            mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "Unnamed molecule"
            if mol_name in molecule_to_xscore:
                xscore = molecule_to_xscore[mol_name]
                if xscore < min_xscore:
                    min_xscore = xscore
                    min_xscore_mol = mol
                    min_xscore_mol_name = mol_name
        
        if min_xscore_mol:
            selected_compounds.append(min_xscore_mol)
            selected_compounds_names.append(min_xscore_mol_name)
            print(f"Cluster {i}: XScore: {min_xscore}, Name: {min_xscore_mol_name}")
            
            # 显示分子图像
            img = Draw.MolToImage(min_xscore_mol, size=(300, 300))
            display(img)  # 确保这里是正确的显示方式

            # 使用分子簇编号和分子名称保存图像
            image_filename = f"Cluster_{i}_{min_xscore_mol_name}.jpg"
            image_path = os.path.join(save_path, image_filename)
            img.save(image_path, dpi=(600, 600))

print("Images saved successfully.")