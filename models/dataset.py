import ast
import os
import json
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import from_smiles

__all__ = ['LBAPDatasetWithSub', 'LBAPDatasetWithChem']

class LBAPDatasetWithSub(Dataset):
    """
    纯净版 Dataset：直接读取 JSON 文件，并使用 PyG 将 SMILES 转换为图
    """
    def __init__(self, ann_file: str, split: str, indices=None):
        super(LBAPDatasetWithSub, self).__init__()
        
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"找不到数据文件: {ann_file}")
            
        print(f"[Dataset] 正在尝试加载大型 JSON 文件: {ann_file}")
        try:
            # 1. 尝试标准方法
            with open(ann_file, 'r') as f:
                full_data = json.load(f)
        except json.JSONDecodeError:
            print("[Dataset] 标准 JSON 加载失败，尝试逐行拼接模式...")
            # 2. 如果标准方法失败，启动备用方案
            content = ""
            with open(ann_file, 'r') as f:
                for line in f:
                    content += line.strip()
            
            try:
                # 拼接完所有行后，再进行一次性解析
                full_data = json.loads(content)
            except json.JSONDecodeError as e:
                # 如果两种方法都失败，就彻底放弃并报错
                print(f"[Dataset] 致命错误：文件 {ann_file} 格式严重损坏，无法解析！")
                raise e
        # ==========================================================
            
        if split not in full_data['split']:
            raise ValueError(f"JSON 文件中没有找到 split: {split}")
            
        records = full_data['split'][split]
        self.indices = list(range(len(records))) if indices is None else list(indices)
        if len(set(self.indices)) != len(self.indices) or any(i < 0 or i >= len(records) for i in self.indices):
            raise ValueError('Invalid or duplicated dataset indices')
        self.data_infos = [records[i] for i in self.indices]
        print(f"[Dataset] 成功加载 {split} 集，共 {len(self.data_infos)} 个分子。")

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        case = self.data_infos[idx]
        smiles = case["smiles"]
        
        # --- 核心转换：将 SMILES 文本直接转为 PyG 的图对象 ---
        data = from_smiles(smiles)
        
        # ===================================================================
        # 🚨 【核心修改一：特征截取与对齐】
        # PyG 会生成多列特征，但 MyGIN 只要第一列 (原子类型/键型)。
        # 必须在这里进行截取，否则 MyGIN 的 Embedding 层会报错！
        # ===================================================================
        
        # 1. 处理主图节点特征 [num_nodes, features] -> [num_nodes, 1]
        # 取第一列 (原子序数) 并确保是长整型。
        # clamp(max=38) 是为了绝对安全：防止有罕见原子的序数超过 38，导致 Embedding(39) 越界崩溃
        data.x = torch.clamp(data.x[:, 0:1].long(), max=38) 
        
        # 2. 处理主图边特征 [num_edges, features] -> [num_edges, 1]
        if data.edge_attr is not None and data.edge_attr.numel() > 0:
            # 取第一列 (键的类型)
            data.edge_attr = torch.clamp(data.edge_attr[:, 0:1].long(), max=9)
        else:
            # 兜底：处理没有化学键的孤立原子
            data.edge_attr = torch.empty((0, 1), dtype=torch.long)
        # ===================================================================
        
        # --- 附加标签信息 ---
        if "cls_label" not in case:
            raise KeyError(f"数据异常！JSON 中找不到 'cls_label' 键！样本信息：{case}")
        data.y = torch.tensor([int(case["cls_label"])], dtype=torch.long)
        
        if "domain_id" not in case:
            raise KeyError(f"数据异常！JSON 中找不到 'domain_id' 键！样本信息：{case}")
        data.domain_id = torch.tensor([int(case["domain_id"])], dtype=torch.long)
        
        # --- 处理论文最核心的“子结构 (Substructure)” ---
        subs_str = case.get('substructure', "{}")
        if isinstance(subs_str, str):
            try:
                subs_list = sorted(ast.literal_eval(subs_str))
            except:
                subs_list = [smiles]
        else:
            subs_list = subs_str
            
        if len(subs_list) == 0:
            subs_list = [smiles]
            
        # 将每一个子结构的 SMILES 也转换为图，并存入主图对象中
        sub_graphs = [from_smiles(sub) for sub in subs_list]
        
        # ===================================================================
        # 🚨 【核心修改二：对所有子结构图执行同样的特征截取】
        # ===================================================================
        for sub_data in sub_graphs:
            sub_data.x = torch.clamp(sub_data.x[:, 0:1].long(), max=38)
            if sub_data.edge_attr is not None and sub_data.edge_attr.numel() > 0:
                sub_data.edge_attr = torch.clamp(sub_data.edge_attr[:, 0:1].long(), max=9)
            else:
                sub_data.edge_attr = torch.empty((0, 1), dtype=torch.long)
        # ===================================================================

        data.subs = sub_graphs
        
        return data

# 保留兼容类
class LBAPDatasetWithChem(LBAPDatasetWithSub):
    def __init__(self, **kwargs):
        super(LBAPDatasetWithChem, self).__init__(**kwargs)
    def __getitem__(self, idx):
        data = super(LBAPDatasetWithChem, self).__getitem__(idx)
        data.smiles = self.data_infos[idx]["smiles"]
        return data

def collate_molecules(items):
    """Preserve variable-length substructure lists without PyG sequence zipping."""
    from torch_geometric.data import Batch
    batch = Batch.from_data_list(items, exclude_keys=['subs'])
    batch.subs = [item.subs for item in items]
    return batch
