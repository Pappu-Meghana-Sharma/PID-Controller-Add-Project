import numpy as np
class ResearchController:
    def __init__(self):
        self.lookup_table = {
            0.027: [12.0, 0.6, 0.01],
            0.045: [15.5, 0.8, 0.02],
            0.060: [18.0, 1.1, 0.03]
        }
        
        
    def get_gains(self, mass):
        keys = sorted(self.lookup_table.keys())
        m_min, m_max = keys[0], keys[-1]
        m = np.clip(mass, m_min, m_max)
        
        for i in range(len(keys)-1):
            if keys[i] <= m <= keys[i+1]:
                m0, m1 = keys[i], keys[i+1]
                g0, g1 = np.array(self.lookup_table[m0]), np.array(self.lookup_table[m1])
                return g0 + (m - m0) * (g1 - g0) / (m1 - m0)
        return self.lookup_table[m_min]
    
    def run_cascaded_control(self,target_angles,curr_angles,curr_rates,gains):
        target_rates=(target_angles-curr_angles)*gains[0]
        torque=(target_rates-curr_rates)*gains[1]
        return torque