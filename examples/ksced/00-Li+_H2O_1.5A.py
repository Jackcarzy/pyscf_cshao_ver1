import numpy
from pyscf import dft, gto, lib, scf

#0 whole system
mol = gto.M(
    verbose = 4,
    atom = '''
    o    0    0.       0.
    h    0    -0.757   0.587
    h    0    0.757    0.587
    li   0    0        -1.5''',
    charge = 1,
    basis = '6-31g')

#1 embedding B (Li+)
molb = gto.M(
verbose = 4,
atom = '''
 x-o    0    0.       0.
 x-h    0    -0.757   0.587
 x-h    0    0.757    0.587
   li   0    0        -1.5''',
charge = 1,
basis = '6-31g')
mfb = dft.RKS(molb)
mfb.xc = 'PBE'
mfb.kernel()

#2 get dm_b, vne_b
dm_b = mfb.make_rdm1()
vne_b = molb.intor('int1e_nuc')

#3 embedded A (H2O)
mola = gto.M(
verbose = 4,
atom = '''
o    0    0.       0.
h    0    -0.757   0.587
h    0    0.757    0.587
x-li   0    0        -1.5''',
basis = '6-31g')
mfa= dft.RKS(mola)
mfa.xc = 'PBE'
mfa.kernel(KSCED=True, vne_b=vne_b, dm_b=dm_b)

#4 just A (H2O)
mola2 = gto.M(
verbose = 4,
atom = '''
o    0    0.       0.
h    0    -0.757   0.587
h    0    0.757    0.587
x-li   0    0        -1.5''',
basis = '6-31g')
mfa2= dft.RKS(mola2)
mfa2.xc = 'PBE'
mfa2.kernel()

#5 get interaction energy (kcal/mol)
Nuc_rep = mol.energy_nuc() - mola2.energy_nuc()- molb.energy_nuc()
Eint = mfa.e_tot - mfa2.e_tot + Nuc_rep 
Eint = Eint*627.503
print( "Nuc_rep:", Nuc_rep)
print( "Embedding interaction energy (kcal/mol) " , Eint)
