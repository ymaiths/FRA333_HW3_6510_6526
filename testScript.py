# file สำหรับตรวจคำตอบ
# ในกรณีที่มีการสร้าง function อื่น ๆ ให้ระบุว่า input-output คืออะไรด้วย
'''
ชื่อ_รหัส(ex: ธนวัฒน์_6461)
1.ชนธัญ_6510
2.ธนพร_6526
'''
import unittest
from FRA333_HW3_6510_6526 import endEffectorJacobianHW3, checkSingularityHW3, computeEffortHW3
import numpy as np

class TestHW3Functions(unittest.TestCase):
    
#===========================================<ตรวจคำตอบข้อ 1>====================================================#
#code here
    def test_jacobian(self):
        q = [0, 0, 0]
        J_e = endEffectorJacobianHW3(q)
        # Expected shape is (6, 3) since we have 3 joints
        self.assertEqual(J_e.shape, (6, 3))
        
#==============================================================================================================#
#===========================================<ตรวจคำตอบข้อ 2>====================================================#
#code here
    def test_singularity(self):
        q = [0, 0, 0]
        self.assertTrue(checkSingularityHW3(q))
        
#==============================================================================================================#
#===========================================<ตรวจคำตอบข้อ 3>====================================================#
#code here
    def test_effort(self):
        q = [0, 0, 0]
        w = np.array([10, 5, 0, 0, 0, 0])
        tau = computeEffortHW3(q, w)
        # Expecting an output with length equal to the number of joints
        self.assertEqual(len(tau), 3)
        
#==============================================================================================================#
if __name__ == "__main__":
    unittest.main()
