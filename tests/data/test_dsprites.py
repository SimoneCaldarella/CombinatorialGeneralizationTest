import unittest
import numpy as np

from displacementae.data.dsprites import DspritesDataset

root = "/home/hamza/datasets/dsprites"

class TestDsprites(unittest.TestCase):

    def test_latents_to_index(self):
        dhandler = DspritesDataset(
            root = root,fixed_in_sampling=[0,1,2],fixed_values=[0,0,5])
        joints1 = [0,0,0,0,0,0]
        idx1 = dhandler.latents_2_index(np.array(joints1))
        self.assertTrue((dhandler._classes[idx1]==joints1).all())
        joints2 = [0,0,0,5,0,0]
        idx2 = dhandler.latents_2_index(np.array(joints2))
        self.assertTrue((dhandler._classes[idx2]==joints2).all())


    def test_get_indices_vary_latents(self):
        dhandler = DspritesDataset(
            root = root,fixed_in_sampling=[0,1,2],fixed_values=[0,0,5])
        
        ret = dhandler.get_indices_vary_latents(vary_latents=[3])

        expected = []
        joint = np.array(
            [0,0,5,0,dhandler.num_latents[4]//2,dhandler.num_latents[5]//2])
        for j in range(dhandler.num_latents[3]):
            joint[3] = j
            expected.append(dhandler.latents_2_index(joint))
        self.assertTrue((ret==expected).all())

        ret2 = dhandler.get_indices_vary_latents(vary_latents=[3,5])
        expected2 = []
        joint2 = np.array([0,0,5,0,dhandler.num_latents[4]//2,0])
        for j1 in range(dhandler.num_latents[3]):
            joint2[3] = j1
            for j2 in range(dhandler.num_latents[5]):
                joint2[5] = j2
                expected2.append(dhandler.latents_2_index(joint2))
        self.assertTrue((ret2==expected2).all())


    def test_multistep(self):
        dhandler = DspritesDataset(
            root = root,fixed_in_sampling=[0,1,2],fixed_values=[0,0,5],
            fixed_in_action=[0,1,2],transitions_on=True,n_transitions=2,
            num_train=200,num_val=30, cyclic_trans=True)
        self.assertEqual(dhandler.n_transitions,2)
        self.assertEqual(dhandler.train_idx.shape,(dhandler.num_train,
                                                   dhandler.n_transitions+1))

        self.assertEqual(dhandler.train_dj.shape,(dhandler.num_train,
                                                  dhandler.n_transitions,
                                                  dhandler.action_shape[0]))
        imgs,_,_ = dhandler[0:50]
        self.assertEqual(imgs.shape,(50,dhandler.n_transitions+1,
                                     *dhandler.in_shape))


    def test_sample_displacement(self):
        dhandler = DspritesDataset(
            root = root,fixed_in_sampling=[0,1,2],fixed_values=[0,0,5])
        d = dhandler._sample_displacement(
            range=[-3,3],n_samples=50,dim=3,dist='uniform')
        self.assertEqual(d.shape , (50,3))
        self.assertTrue(d.max() <= 3)
        self.assertTrue(d.min() >= -3 )
        
        dim = 3
        n_samples = 50
        low,high=-3,3
        d = dhandler._sample_displacement(
            range=[low,high],n_samples=n_samples,dim=dim,dist='disentangled')
        self.assertEqual(d.shape , (50,3))
        self.assertTrue(d.max() <= high )
        self.assertTrue(d.min() >= low )
        # A disentangled d, at most one non zero
        self.assertTrue(((d==0).sum(axis = 1) >= dim -1).all())


if __name__ == '__main__':
    unittest.main()