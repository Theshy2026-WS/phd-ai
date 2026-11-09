import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

import fusion_overnight_delivery as delivery


class CombinedDeliveryTests(unittest.TestCase):
    def fixture(self, root, status='complete'):
        queue = root/'queue'; queue.mkdir()
        children = {}
        for name in delivery.CHILDREN:
            suite = root/name; suite.mkdir()
            children[name] = suite
            (suite/'training').mkdir(); (suite/'feature_analysis').mkdir()
            (suite/'completion.json').write_text(json.dumps(dict(status='complete',
                outcomes=dict(training=dict(status='complete'), feature_analysis=dict(status='complete')))))
            (suite/'training/report.md').write_text('# Training results\n')
            (suite/'training/epochs.csv').write_text('epoch,accuracy\n1,0.65\n')
            (suite/'feature_analysis/summary.json').write_text('{"accuracy":0.65}')
        (queue/'queue.json').write_text(json.dumps(dict(children=[dict(name=name,suite=str(path),core=name+'.py')
            for name,path in children.items()])))
        (queue/'completion.json').write_text(json.dumps(dict(status=status)))
        (queue/'run.log').write_text('Both core stages have ended\n')
        return queue, children

    @staticmethod
    def package():
        return dict(full=Path('combined.zip'), compact=None, attachment=Path('combined.zip'))

    def test_merges_both_studies_as_files_without_nested_archives_or_secrets(self):
        with tempfile.TemporaryDirectory() as temp:
            queue, children = self.fixture(Path(temp))
            training = children['pairwise']/'training'
            for name in ('weights.pt','probabilities.npz','source.py','old.zip','email.txt','.env'):
                (training/name).write_text('excluded')
            (training/'source_snapshot').mkdir()
            (training/'source_snapshot/config.json').write_text('{"not_a_report":true}')
            (queue/'credentials.json').write_text('{"secret":"exclude"}')
            bundle = delivery.bundle_results(queue)
            with zipfile.ZipFile(bundle['attachment']) as archive:
                names = archive.namelist()
                self.assertIsNone(archive.testzip())
                self.assertIn('pairwise/training_analysis/epochs.csv', names)
                self.assertIn('hcsi/feature_analysis/summary.json', names)
                self.assertIn('queue/queue.json', names)
                self.assertIn('delivery_manifest.json', names)
                for name in names:
                    self.assertIn(Path(name).suffix, delivery.ALLOWED_SUFFIXES)
                    self.assertFalse(any(token in name for token in ('weights.pt','probabilities.npz',
                        'source.py','old.zip','email.txt','.env','credentials.json','source_snapshot')))

    def test_compaction_removes_per_image_csv_then_png_but_retains_aggregate_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            queue, children = self.fixture(Path(temp))
            reports = children['pairwise']/'feature_analysis'
            (reports/'per_image.csv').write_text('image_id,prediction\n'+os.urandom(20000).hex()+',1\n')
            (reports/'gallery.png').write_bytes(os.urandom(20000))
            (reports/'paired_summary.json').write_text('{"corrected":455,"harmed":390}')
            with patch.object(delivery,'ATTACHMENT_LIMIT',6000):
                bundle = delivery.bundle_results(queue)
            self.assertEqual(bundle['compact_level'],2)
            with zipfile.ZipFile(bundle['attachment']) as archive:
                names = archive.namelist()
                self.assertNotIn('pairwise/feature_analysis/per_image.csv',names)
                self.assertNotIn('pairwise/feature_analysis/gallery.png',names)
                self.assertIn('pairwise/feature_analysis/paired_summary.json',names)

    def test_oversized_reports_retained_locally_without_oversized_attachment(self):
        with tempfile.TemporaryDirectory() as temp:
            queue, children = self.fixture(Path(temp))
            (children['hcsi']/'feature_analysis/large.json').write_text(json.dumps(dict(values=os.urandom(12000).hex())))
            with patch.object(delivery,'ATTACHMENT_LIMIT',500):
                bundle = delivery.bundle_results(queue)
            self.assertIsNone(bundle['attachment'])
            self.assertTrue(bundle['full'].is_file()); self.assertTrue(bundle['compact'].is_file())

    def test_missing_child_has_explicit_partial_markers(self):
        with tempfile.TemporaryDirectory() as temp:
            queue, children = self.fixture(Path(temp))
            missing = Path(temp)/'not_started'
            bundle = delivery.bundle_results(queue,dict(children,hcsi=missing))
            with zipfile.ZipFile(bundle['attachment']) as archive:
                evidence = json.loads(archive.read('hcsi/training_analysis/analysis_bundle.json'))
                self.assertFalse(evidence['directory_exists'])
                self.assertEqual(evidence['report_file_count'],0)
                self.assertIn('partial evidence',evidence['note'])

    def test_rejects_duplicate_or_path_injection_child_names(self):
        with tempfile.TemporaryDirectory() as temp:
            queue, children = self.fixture(Path(temp))
            with self.assertRaises(ValueError):
                delivery.bundle_results(queue,{'pairwise':children['pairwise'],'../outside':children['hcsi']})
            with self.assertRaises(ValueError):
                delivery.bundle_results(queue,dict(pairwise=children['pairwise'],hcsi=children['pairwise']))
            value = dict(children=[dict(name='pairwise',suite=str(children['pairwise']))]*2)
            (queue/'queue.json').write_text(json.dumps(value))
            with self.assertRaises(ValueError):delivery.child_suites(queue)

    def test_rejects_symlink_root_and_skips_symlink_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); queue,children=self.fixture(root)
            secret=root/'outside_secret.txt';secret.write_text('secret')
            link=children['pairwise']/'training/link.txt'
            try:link.symlink_to(secret)
            except OSError:self.skipTest('Symlinks unavailable on this platform')
            bundle=delivery.bundle_results(queue)
            with zipfile.ZipFile(bundle['attachment']) as archive:
                self.assertNotIn('pairwise/training_analysis/link.txt',archive.namelist())
            root_link=root/'suite_link';root_link.symlink_to(children['hcsi'],target_is_directory=True)
            with self.assertRaises(ValueError):delivery.bundle_results(queue,dict(children,hcsi=root_link))

    def test_success_status_uses_children_not_nonexistent_parent_training_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            queue,_=self.fixture(Path(temp))
            self.assertFalse((queue/'training').exists())
            self.assertEqual(delivery.queue_status(queue,0),'SUCCESS')

    def test_partial_and_cooperative_budget_statuses_are_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            queue,children=self.fixture(Path(temp))
            (children['hcsi']/'completion.json').write_text('{"status":"failed"}')
            self.assertEqual(delivery.queue_status(queue,0),'PARTIAL / CHILD INCOMPLETE')
            (queue/'completion.json').write_text('{"status":"budget_exhausted"}')
            self.assertEqual(delivery.queue_status(queue,2),'PARTIAL / TIME LIMIT')
            self.assertEqual(delivery.queue_status(queue,1,timed_out=True),'PARTIAL / TIME LIMIT')

    def test_one_combined_email_then_accounting_then_managed_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            queue,_=self.fixture(Path(temp));events=[]
            with patch.object(delivery,'bundle_results',side_effect=lambda *a:(events.append('package') or self.package())), \
                    patch.object(delivery,'send_mail',side_effect=lambda *a,**k:events.append('email')) as mail, \
                    patch.object(delivery,'stop_workbench',side_effect=lambda *a:events.append('stop')) as stop:
                complete=delivery.finish(queue,queue/'run.log','me@example.com',0,'managed-url',
                    before_stop=lambda:events.append('accounting'))
            self.assertTrue(complete);mail.assert_called_once();stop.assert_called_once_with('managed-url',False)
            self.assertEqual(events,['package','email','accounting','stop'])
            record=json.loads((queue/'delivery_status.json').read_text())
            self.assertTrue(record['stop_requested']);self.assertTrue(record['email_accepted'])
            self.assertFalse(record['managed_stop_confirmed'])

    def test_failed_training_still_emails_and_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            queue,_=self.fixture(Path(temp),status='failed');events=[]
            with patch.object(delivery,'bundle_results',return_value=self.package()), \
                    patch.object(delivery,'send_mail',side_effect=lambda *a,**k:events.append('mail')) as mail, \
                    patch.object(delivery,'stop_workbench',side_effect=lambda *a:events.append('stop')):
                self.assertFalse(delivery.finish(queue,queue/'run.log','me@example.com',7,'url'))
            self.assertIn('FAILED / INTERRUPTED',mail.call_args.args[1])
            self.assertEqual(events,['mail','stop'])

    def test_package_failure_emails_without_attachment_and_still_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            queue,_=self.fixture(Path(temp))
            with patch.object(delivery,'bundle_results',side_effect=OSError('disk full')), \
                    patch.object(delivery,'send_mail') as mail,patch.object(delivery,'stop_workbench') as stop:
                self.assertFalse(delivery.finish(queue,queue/'run.log','me@example.com',0,'url'))
            self.assertIsNone(mail.call_args.kwargs['attachment']);stop.assert_called_once()
            self.assertIn('PACKAGE FAILED',mail.call_args.args[1])

    def test_mail_failure_retries_then_stops_after_all_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            queue,_=self.fixture(Path(temp));events=[]
            def fail(*args,**kwargs):events.append('mail');raise RuntimeError('offline')
            with patch.object(delivery,'bundle_results',return_value=self.package()), \
                    patch.object(delivery,'send_mail',side_effect=fail),patch.object(delivery.time,'sleep'), \
                    patch.object(delivery,'stop_workbench',side_effect=lambda *a:events.append('stop')):
                self.assertFalse(delivery.finish(queue,queue/'run.log','me@example.com',0,'url'))
            self.assertEqual(events,['mail','mail','mail','stop'])
            self.assertFalse(json.loads((queue/'delivery_status.json').read_text())['email_accepted'])

    def test_budget_or_delivery_record_failure_cannot_prevent_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            queue,_=self.fixture(Path(temp));events=[]
            def fail_callback():events.append('accounting');raise OSError('ledger error')
            with patch.object(delivery,'bundle_results',return_value=self.package()), \
                    patch.object(delivery,'send_mail'), \
                    patch.object(delivery,'_record_delivery',side_effect=OSError('record error')), \
                    patch.object(delivery,'stop_workbench',side_effect=lambda *a:events.append('stop')):
                with self.assertRaises(OSError):
                    delivery.finish(queue,queue/'run.log','me@example.com',0,'url',before_stop=fail_callback)
            self.assertEqual(events,['accounting','stop'])

    def test_stop_failure_is_not_reported_as_confirmed_shutdown(self):
        with tempfile.TemporaryDirectory() as temp:
            queue,_=self.fixture(Path(temp))
            with patch.object(delivery,'bundle_results',return_value=self.package()), \
                    patch.object(delivery,'send_mail'), \
                    patch.object(delivery,'stop_workbench',side_effect=RuntimeError('stop rejected')):
                with self.assertRaisesRegex(RuntimeError,'stop rejected'):
                    delivery.finish(queue,queue/'run.log','me@example.com',0,'url',fallback=True)
            record=json.loads((queue/'delivery_status.json').read_text())
            self.assertTrue(record['stop_requested']);self.assertFalse(record['managed_stop_confirmed'])
            self.assertEqual(record['stop_error_type'],'RuntimeError')

    def test_optional_expired_budget_skips_mail_retry_but_still_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            queue,_=self.fixture(Path(temp))
            budget=dict(total_seconds=36000,total_deadline_monotonic=time.monotonic()-1)
            with patch.object(delivery,'bundle_results',return_value=self.package()), \
                    patch.object(delivery,'send_mail') as mail,patch.object(delivery,'stop_workbench') as stop:
                self.assertFalse(delivery.finish(queue,queue/'run.log','me@example.com',0,'url',budget=budget))
            mail.assert_not_called();stop.assert_called_once()


if __name__=='__main__':unittest.main()
