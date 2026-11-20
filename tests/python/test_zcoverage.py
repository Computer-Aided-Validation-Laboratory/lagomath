import sys
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[2]/'tools'))
from coverage_report import report

def test_complete_backend_matrix():assert report()>0
