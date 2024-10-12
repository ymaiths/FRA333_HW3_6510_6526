# file สำหรับเขียนคำตอบ
# ในกรณีที่มีการสร้าง function อื่น ๆ ให้ระบุว่า input-output คืออะไรด้วย
'''
ชื่อ_รหัส(ธนวัฒน์_6461)
1.ชนธัญ_6510
2.ธนพร_6526
'''
#=============================================<คำตอบข้อ 1>======================================================#
#code here
import numpy as np
import HW3_utils as utils
def endEffectorJacobianHW3(q:list[float])->list[float]:
    # Get forward kinematics results
    R, P, R_e, p_e = utils.FKHW3(q)

    # Number of joints
    n = 3

    # Initialize Jacobian matrices
    J_v = np.zeros((3, n))  # Linear velocity part
    J_w = np.zeros((3, n))  # Angular velocity part

    # Base frame Z-axis (always along Z in base frame)
    z0 = np.array([0, 0, 1])

    # Position of the end-effector
    p_e = P[:, -1]

    # Joint 1 contribution (Revolute)
    p_0 = np.array([0, 0, 0])  # Position of joint 1 (base)
    J_v[:, 0] = np.cross(z0, (p_e - p_0))  # Linear velocity
    J_w[:, 0] = z0  # Angular velocity

    # Joint 2 contribution (Revolute)
    z1 = R[:, 2, 0]  # Z-axis of frame 1
    p_1 = P[:, 0]  # Position of joint 2
    J_v[:, 1] = np.cross(z1, (p_e - p_1))  # Linear velocity
    J_w[:, 1] = z1  # Angular velocity

    # Joint 3 contribution (Revolute)
    z2 = R[:, 2, 1]  # Z-axis of frame 2
    p_2 = P[:, 1]  # Position of joint 3
    J_v[:, 2] = np.cross(z2, (p_e - p_2))  # Linear velocity
    J_w[:, 2] = z2  # Angular velocity

    # Construct the full Jacobian (6xN)
    J_e = np.vstack((J_v, J_w))

    return J_e
#==============================================================================================================#
#=============================================<คำตอบข้อ 2>======================================================#
#code here
def checkSingularityHW3(q:list[float])->bool:
    # Get the Jacobian matrix for the given joint configuration
    J_e = endEffectorJacobianHW3(q)

    # Calculate the determinant of J_v (top 3 rows for linear velocity)
    det_J_v = np.linalg.det(J_e[:3, :])

    # Check if determinant is close to zero (singular)
    if abs(det_J_v) < 0.001:
        return True  # Near singularity
    else:
        return False  # Not near singularity

#==============================================================================================================#
#=============================================<คำตอบข้อ 3>======================================================#
#code here
def computeEffortHW3(q:list[float], w:list[float])->list[float]:
    # Get the Jacobian matrix for the given joint configuration
    J_e = endEffectorJacobianHW3(q)

    # Calculate the joint torques/efforts using the Jacobian transpose
    tau = np.dot(J_e.T, w)

    return tau
#==============================================================================================================#