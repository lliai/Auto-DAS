from diswotv2.searchspace.instincts.linear import LinearInstinct
from diswotv2.searchspace.interactions.parallel import ParaInteraction

instinct = LinearInstinct()
interaction = ParaInteraction()

print(' == instinct ==')
print(f'before load: {instinct}')
instinct.update_genotype(
    'INPUT:(virtual_grad)UNARY:|abslog|sigmoid|normalized_sum|invert|')
print(f'after load: {instinct}')

print(' == interaction ==')
print(f'before load: {interaction}')
interaction.update_alleletype(
    "ALLELE# in:['k3']~       trans:['trans_drop', 'trans_multi_scale_r4', 'trans_pow2']~     weig:['w100_teacher_student']~  dist:['l1_loss']"
)
print(f'after load: {interaction}')

# s0 0.9096
#  * best_interaction:  * ALLELE# in:['k2']~       trans:['trans_relu', 'trans_sigmoid', 'trans_abs']~     weig:['w100_teacher_student']~  dist:['kl_T8']
#  * best_instinct: INPUT:(grad)UNARY:|logsoftmax|no_op|frobenius_norm|normalized_sum|

# nb201 0.7469
#  * best_interaction:  * ALLELE# in:['k2']~       trans:['trans_mish', 'trans_pow2', 'trans_softmax_N']~  weig:['w100_teacher_student']~  dist:['multiply']
#  * best_instinct: INPUT:(grad)UNARY:|logsoftmax|no_op|tanh|l1_norm|

# nb101 0.6582
#  * best_interaction:  * ALLELE# in:['k3']~       trans:['trans_drop', 'trans_multi_scale_r4', 'trans_pow2']~     weig:['w100_teacher_student']~  dist:['l1_loss']
#  * best_instinct: INPUT:(virtual_grad)UNARY:|abslog|sigmoid|normalized_sum|invert|
