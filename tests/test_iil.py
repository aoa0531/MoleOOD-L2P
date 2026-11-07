"""Run from project root: python -m unittest discover -s tests -v."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data
from models import Framework
from models.dataset import collate_molecules
from models.prompt_pool import PromptPool, step_selected_rows
from prepare_iil_split import make_split, validate_split


def graph(n):
    edges = [(i, i+1) for i in range(n-1)] + [(i+1, i) for i in range(n-1)]
    return Data(x=torch.full((n, 1), 6, dtype=torch.long),
                edge_index=torch.tensor(edges, dtype=torch.long).T.contiguous(),
                edge_attr=torch.ones(len(edges), 1, dtype=torch.long))


def batch():
    a, b = graph(3), graph(2)
    a.subs, b.subs = [graph(2), graph(3)], [graph(2)]
    return collate_molecules([a, b])


class PromptTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)

    def test_zero_prompt_and_legacy_fusion_without_labels(self):
        data = batch()
        model = Framework(8, 8, 2).eval()
        with torch.no_grad():
            full = model.base_model(data.x, data.edge_index, data.edge_attr, data.batch)
            subs = []
            for group in data.subs:
                features = [model.sub_model(s.x, s.edge_index, s.edge_attr,
                            torch.zeros(len(s.x), dtype=torch.long))[0] for s in group]
                subs.append(torch.stack(features).mean(0))
            expected = model.predictor(full + torch.stack(subs))
            baseline = model(data)
        torch.testing.assert_close(expected, baseline)
        model.enable_prompts(3)
        with torch.no_grad():
            model.prompt_pool.prompts.zero_()
        torch.testing.assert_close(baseline, model(data))

    def test_gradients_and_frozen_state(self):
        model = Framework(8, 8, 2, dropout=0, prompt_pool_size=3)
        model.train()
        self.assertFalse(model.base_model.training)
        self.assertFalse(model.sub_model.training)
        before = copy.deepcopy(model.base_model.state_dict())
        result = model(batch(), return_aux=True)
        torch.nn.functional.cross_entropy(result['logits'], torch.tensor([0, 1])).backward()
        self.assertGreater(model.prompt_pool.prompts.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.base_model.parameters()))
        model.zero_grad(set_to_none=True)
        model(batch(), return_aux=True)['key_loss'].backward()
        self.assertGreater(model.prompt_pool.keys.grad.abs().sum().item(), 0)
        for k, v in before.items():
            self.assertTrue(torch.equal(v, model.base_model.state_dict()[k]))

    def test_node_assignment(self):
        model = Framework(8, 8, 2).eval()
        data = batch()
        captured = []
        handle = model.base_model.gnn_layers[0].register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone()))
        args = (data.x, data.edge_index, data.edge_attr, data.batch)
        model.base_model(*args)
        prompt = torch.stack([torch.ones(8), torch.full((8,), 3.)])
        model.base_model(*args, graph_prompt=prompt)
        handle.remove()
        torch.testing.assert_close(captured[1]-captured[0], prompt[data.batch])

    def test_inactive_rows_and_momentum(self):
        pool = PromptPool(3, 8)
        optimizer = torch.optim.Adam(pool.parameters(), lr=.01, weight_decay=.01)
        for selected in (0, 1):
            optimizer.zero_grad(set_to_none=True)
            (pool.prompts[selected].sum()+pool.keys[selected].sum()).backward()
            before = [p.detach().clone() for p in pool.parameters()]
            old_state = [{k:v.clone() for k,v in optimizer.state.get(p, {}).items()
                          if torch.is_tensor(v) and v.shape == p.shape} for p in pool.parameters()]
            step_selected_rows(optimizer, pool, torch.tensor([selected]))
            inactive = [i for i in range(3) if i != selected]
            for p, original, state in zip(pool.parameters(), before, old_state):
                self.assertTrue(torch.equal(p[inactive], original[inactive]))
                for key, value in state.items():
                    self.assertTrue(torch.equal(optimizer.state[p][key][inactive], value[inactive]))

    def test_checkpoint_roundtrip(self):
        model = Framework(8, 8, 2, prompt_pool_size=3).eval()
        restored = Framework(8, 8, 2, prompt_pool_size=3).eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(batch()), restored(batch()))

    def test_split_repeatability_and_duplicates(self):
        rows = [{'smiles': f'molecule-{i}', 'cls_label': i % 2} for i in range(60)]
        rows.append(rows[0].copy())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.json'
            path.write_text(json.dumps({'split': {'train': rows}}))
            first = make_split(path)
            self.assertEqual(first, make_split(path))
            validate_split(first, path)
            self.assertEqual(first['duplicate_groups'], 1)

    def test_end_to_end_and_exact_stage_resume(self):
        import train_iil
        rows = [{'smiles': 'C'*(i+2), 'cls_label': i % 2, 'domain_id': i,
                 'substructure': "{'CC', 'CCC'}" if i % 2 else "{'CC'}"}
                for i in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data.json'
            data.write_text(json.dumps({'split': {'train': rows,
                             'iid_val': rows[:4], 'ood_test': rows[-4:]}}))
            manifest = make_split(data, fractions=(.5, .25, .25))
            split = root/'split.json'
            split.write_text(json.dumps(manifest))
            cfg = json.loads((train_iil.ROOT/'configs/iil_scaffold.json').read_text())
            cfg.update(data_path=str(data), split_path=str(split), output_dir=str(root/'run'),
                       emb_dim=8, pool_size=3, batch_size=4, num_domain=2,
                       assistant_epochs=1, initial_epochs=1, incremental_epochs=1,
                       eval_splits=['iid_val', 'ood_test'])
            config = root/'config.json'
            config.write_text(json.dumps(cfg))
            with patch('sys.argv', ['train_iil.py', '--config', str(config)]):
                train_iil.main()
            original = torch.load(root/'run/stage_2.pt', weights_only=False)
            with patch('sys.argv', ['train_iil.py', '--config', str(config),
                                    '--resume', str(root/'run/stage_1.pt')]):
                train_iil.main()
            resumed = torch.load(root/'run/stage_2.pt', weights_only=False)
            for key, value in original['main'].items():
                self.assertTrue(torch.equal(value, resumed['main'][key]), key)
            self.assertEqual(original['history'], resumed['history'])
            with patch('sys.argv', ['evaluate.py', '--iil-checkpoint',
                       str(root/'run/stage_2.pt'), '--splits', 'ood_test']):
                train_iil.evaluation_main()

            import train_head_only
            with patch('sys.argv', ['train_head_only.py', '--source',
                       str(root/'run/stage_0.pt'), '--output-dir', str(root/'head')]):
                train_head_only.main()
            source = torch.load(root/'run/stage_0.pt', weights_only=False)
            baseline = torch.load(root/'head/stage_2.pt', weights_only=False)
            self.assertFalse(baseline['prompted'])
            self.assertFalse(any(k.startswith('prompt_pool.') for k in baseline['main']))
            for k, v in baseline['main'].items():
                if not k.startswith('predictor.'):
                    self.assertTrue(torch.equal(v, source['main'][k]), k)
            self.assertTrue(any(not torch.equal(v, source['main'][k])
                                for k, v in baseline['main'].items()
                                if k.startswith('predictor.')))
            with patch('sys.argv', ['train_head_only.py', '--resume',
                                    str(root/'head/stage_1.pt')]):
                train_head_only.main()
            head_resumed = torch.load(root/'head/stage_2.pt', weights_only=False)
            for k, v in baseline['main'].items():
                self.assertTrue(torch.equal(v, head_resumed['main'][k]), k)
            self.assertEqual(baseline['history'], head_resumed['history'])


if __name__ == '__main__':
    unittest.main()
