import copy
import importlib
import unittest
import numpy as np

class ReportTests(unittest.TestCase):
    def setUp(self):
        self.m = importlib.import_module('tools.report_adapter_study')
        self.rows = []
        self.predictions = {}
        for seed in (42,43,44):
            self.predictions[str(seed)] = {}
            for arm in ('correct','reference'):
                self.predictions[str(seed)][arm] = []
        for j in range(8):
            for i in range(32):
                row = dict(token=f't{j}_{i}', route=f'group{j}_0_100', cohort='holdout', split='val', timestamp_us=i,
                           command=1, ego=[2.,0.,0.,0.], trajectory_gt=np.zeros((8,3)).tolist())
                self.rows.append(row)
                for seed in (42,43,44):
                    for arm in ('correct','reference'):
                        for mode in ('correct','off','reference','shuffled'):
                            error = 1. if mode=='off' else (0.4 if arm=='correct' and mode=='correct' else .5)
                            pred = np.zeros((8,3)); pred[:,0] = error
                            self.predictions[str(seed)][arm].append(dict(token=row['token'], route=row['route'], cohort='holdout',
                                command=1, speed_mps=2., mode=mode, source_token=None if mode=='off' else 'r0',
                                prediction=pred.tolist(), target=row['trajectory_gt']))

    def test_complete_assembly_primary_is_equal_log_then_equal_seed(self):
        windows = self.m.assemble_windows(self.rows,self.predictions,cohort='holdout')
        stats = self.m.primary_analysis(windows)
        self.assertEqual(len(windows),256)
        np.testing.assert_allclose(stats['paired_gain_matrix_m'],.1,atol=1e-12)
        self.assertTrue(stats['passes_primary_gate'])
        self.assertEqual(stats['positive_log_count'],8)
        self.assertEqual(stats['replicates'],20000)

    def test_missing_condition_rejected(self):
        self.predictions['43']['reference'].pop()
        with self.assertRaisesRegex(ValueError,'condition|coverage'):
            self.m.assemble_windows(self.rows,self.predictions,cohort='holdout')

    def test_different_off_baseline_rejected(self):
        self.predictions['44']['reference'][1]['prediction'][0][0] += .01
        with self.assertRaisesRegex(ValueError,'[Oo]ff'):
            self.m.assemble_windows(self.rows,self.predictions,cohort='holdout')

    def test_target_change_rejected(self):
        self.predictions['42']['correct'][0]['target'] = np.ones((8,3)).tolist()
        with self.assertRaisesRegex(ValueError,'target|Target'):
            self.m.assemble_windows(self.rows,self.predictions,cohort='holdout')

    def test_duplicate_prediction_rejected(self):
        self.predictions['42']['correct'].append(copy.deepcopy(self.predictions['42']['correct'][0]))
        with self.assertRaisesRegex(ValueError,'[Dd]uplicate|coverage'):
            self.m.assemble_windows(self.rows,self.predictions,cohort='holdout')

    def test_metric_drift_rejected(self):
        records = self.predictions['42']['correct']
        saved = {mode:self.m.pose_metrics([r['prediction'] for r in records if r['mode']==mode],
                    [r['target'] for r in records if r['mode']==mode]) for mode in self.m.MODES}
        self.assertLessEqual(self.m.recompute_metric_error(records,saved),1e-12)
        saved['correct']['ADE']+=.001
        with self.assertRaisesRegex(ValueError,'metric'):
            self.m.recompute_metric_error(records,saved)

    def test_recording_group_leakage_rejected(self):
        old = [dict(token='old',route='group0_101_200',cohort='train',split='train')]
        with self.assertRaisesRegex(ValueError,'recording|boundary'):
            self.m.validate_cohorts(old+self.rows)

    def test_partial_actual_replay_cannot_pass(self):
        replay={key:dict(passed=True,max_abs_difference=0) for key in ('fusion','optimizer','scheduler','rng','reference_prediction')}
        check=dict(conditions=1024,full_holdout_predictions=dict(passed=True,max_abs_difference=0),
                   actual_step_replay=replay,before=511,after=512)
        proof=dict(passed=True,new_network_bytes=0,comparisons={str(seed):{arm:copy.deepcopy(check) for arm in ('correct','reference')} for seed in (42,43,44)})
        self.m.validate_replay(proof)
        del proof['comparisons']['44']['correct']['actual_step_replay']['optimizer']
        with self.assertRaisesRegex(ValueError,'replay'):
            self.m.validate_replay(proof)

    def test_nonnative_cold_deployment_cannot_pass(self):
        checks=[dict(seed=seed,token=token,exact=True,max_abs_difference=0,native_processor_exact=True,
                original_head_calls=1,visual_tokens=[100]*4) for seed in (42,43,44) for token in ('old','new')]
        proof=dict(passed=True,no_slow_or_reader=True,models_constructed=dict(Fast=1,Slow=0,Reader=0),new_network_bytes=0,checks=checks)
        self.m.validate_deployment(proof,['old','new'])
        proof['checks'][4]['native_processor_exact']=False
        with self.assertRaisesRegex(ValueError,'processor'):
            self.m.validate_deployment(proof,['old','new'])

    def test_all_source_tokens_follow_locked_interventions(self):
        reference='old_train_reference'
        donors={r['token']:self.rows[(i+128)%256]['token'] for i,r in enumerate(self.rows)}
        for pair in self.predictions.values():
            for records in pair.values():
                for r in records:
                    r['source_token'] = {'off':None,'correct':r['token'],'reference':reference,'shuffled':donors[r['token']]}[r['mode']]
        self.m.validate_source_tokens(self.predictions,donors,reference)
        self.predictions['43']['reference'][3]['source_token']='wrong'
        with self.assertRaisesRegex(ValueError,'source|donor'):
            self.m.validate_source_tokens(self.predictions,donors,reference)

    def test_zero_initial_residual_blocks_upstream_gradients_only_on_first_update(self):
        probes = [dict(step=step, gradients={'residual.weight':dict(finite=True,nonzero=True),
                        'combine.weight':dict(finite=True,nonzero=step!=1)}) for step in (1,2,128,512)]
        self.m.validate_gradient_probes(probes)
        probes[-1]['gradients']['combine.weight']['nonzero']=False
        with self.assertRaisesRegex(ValueError,'gradient'):
            self.m.validate_gradient_probes(probes)

    def test_portable_removes_embedded_absolute_paths(self):
        value = self.m.portable({'path':'/private/foo/x.pt','error':'failed at /private/foo/x.pt','url':'https://host/a'})
        self.assertNotIn('/private/',str(value))
        self.assertEqual(value['url'],'https://host/a')

if __name__=='__main__': unittest.main()
