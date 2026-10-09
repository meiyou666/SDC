1. computing overhead
Bert:
Attention Mechanism Overhead:  0.16686546259406831                                                                        
Training Overhead:  0.02391220952910309
gpt2:
Attention Mechanism Overhead:  0.358206374509681                       
Training Overhead:  0.12417662657818716 
roberta:
Attention Mechanism Overhead:  0.05456632359861377                                                                        
Training Overhead:  0.04392132284383148


2. Detection and Correction Rating 
grep -Ec '\[(col|row) check\].*(error detected|INF detected|NAN detected|chk inf error detected)' fixed_pos0.txt
863
grep -Ec 'Multi|INPUTS|Invalid line number' logs/gpt2_fixed_pos0.log
9
correction rating = 98.967%
detection rating = 92%

3. Adaptive ABFT Detection Frequencies
1) mode2_output:
Attention Mechanism Overhead:  0.17360705327661427                     
* Training Overhead:  0.05734339203229054                                
ATTNChecker Loss:  0.5328999999999999                                  
no ATTNChecker Loss:  0.5328999999999999

2) mode1_output:
Attention Mechanism Overhead:  0.3188771996499093                      
* Training Overhead:  0.12654867007134754                                
ATTNChecker Loss:  0.5328999999999999                                  
no ATTNChecker Loss:  0.5328999999999999

4. Recovery Overhead
1) checkpoint:
output:(The loss here is the 1-step training loss)
Attention Mechanism Overhead:  0.358206374509681                       
Training Overhead:  0.12417662657818716                                
ATTNChecker Loss:  0.5328999999999999                                  
no ATTNChecker Loss:  0.5328999999999999

output:(test the save and load time of Checkpointing of a model)
* Overhead of Checkpointing:  22.754563590047233

2) correction recovery overhead
output:
{'loss': 0.1948, 'learning_rate': 0.0, 'epoch': 0.0}                   
{'train_runtime': 0.4248, 'train_samples_per_second': 18.83, 'train_ste
ps_per_second': 2.354, 'train_loss': 0.19479024410247803, 'epoch': 0.0}
* Time:  0.5386708320002072                                              
Loss:  0.1948

5. Different Batch-size Expriments
batch_size = 1
Attention Mechanism Overhead:  -0.014795321246296922                   
Training Overhead:  0.18021543722703862                                
ATTNChecker Loss:  0.008200000000000002                                
no ATTNChecker Loss:  0.0192            
                      
batch_size = 2                                                    
Attention Mechanism Overhead:  -0.15820624761122273                    
Training Overhead:  -0.11402693524158884                               
ATTNChecker Loss:  0.007599999999999998                                
no ATTNChecker Loss:  0.0106                                           
                
batch_size = 4                             
Attention Mechanism Overhead:  -0.061314452195778024                   
Training Overhead:  -0.03466487481263959                               
ATTNChecker Loss:  0.0063                                              
no ATTNChecker Loss:  0.005400000000000001                             
            
batch_size = 8                                   
Attention Mechanism Overhead:  0.14447428254048392                     
Training Overhead:  0.12366836753376505                                
ATTNChecker Loss:  0.5328999999999999                                  
no ATTNChecker Loss:  0.5328999999999999                               
                  
batch_size = 16                             
Attention Mechanism Overhead:  -0.08521124187890557                    
Training Overhead:  0.04070085540252237                                
ATTNChecker Loss:  1.2513000000000003                                  
no ATTNChecker Loss:  1.2513000000000003                               
                      
batch_size = 32                        
Attention Mechanism Overhead:  0.4817055180875307                    
Training Overhead:  0.19675984220289766                                
ATTNChecker Loss:  3.1664999999999996                                  
no ATTNChecker Loss:  3.1664999999999996

6. Custom Optimized Encoder Kernel
1) V4:
Q empty                                                                
K empty                                                                
V empty                                                                
AS empty                                                               
CL empty                                                               
OUT empty                                                              
* preparation avg_ms=0.168458 n=144                       
cpy empty                                                              
BGemmCorrect empty

2) V3:
Q empty                                                                
K empty                                                                
V empty                                                                
AS empty                                                               
CL empty                                                               
OUT empty                                                              
* preparation avg_ms=0.427071 n=144                                      
cpy empty                                                              
BGemmCorrect empty


