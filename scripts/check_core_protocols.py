"""Run existing lightweight contract tests without Isaac, weights or pytest."""
from pathlib import Path
import inspect,runpy,sys,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
FILES=['test_skill_adapter.py','test_task_sequence.py','test_locomotion_shadow.py',
       'test_locomotion_shadow_verification.py','test_skill_sequence_protocol.py',
       'test_skill_watchdog_protocol.py','test_skill_sequence_retest_metrics.py']

def main():
    if not __debug__:raise RuntimeError('Run normally: these existing tests include Python asserts.')
    suite=unittest.TestSuite()
    for name in FILES:
        namespace=runpy.run_path(str(ROOT/'tests'/name))
        for key,obj in namespace.items():
            if inspect.isclass(obj) and issubclass(obj,unittest.TestCase) and obj is not unittest.TestCase:
                suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(obj))
            elif key.startswith('test_') and inspect.isfunction(obj):
                suite.addTest(unittest.FunctionTestCase(obj,description=name+':'+key))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
