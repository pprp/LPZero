from lpzero.structures.tree import TreeStructure

struct1 = TreeStructure()
genotype1 = struct1.genotype
print('geno1:', genotype1)
print('struct1', struct1)
print('struct1', struct1._repr_geno)

# recover based on the genotype
struct2 = TreeStructure()
genotype2 = struct2.genotype
print('geno2:', genotype2)
print('struct2', struct2)

# recover from genotype1
struct2.genotype = genotype1
print('recovered:', struct2.genotype)
print('struct2', struct2)
