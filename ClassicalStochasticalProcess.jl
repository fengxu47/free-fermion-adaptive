#### Free fermions under adaptive quantum dynamics ####
#### Classical stochastic process
# 1, Given an initial state, represented by bitstrings

# 2, Odd-layer evolution, if f_{2j-1}=1 or f_{2j}=1, update  01 -> 10 or 01 -> 01 is equal probability, the same for 10, 00 , 11
#     then set f_{2j-1}=f_{2j}=1
# 3, Odd-layer measurement and feedback, with probability p, 
# measure the occupation numbers on two neighboring sites (n_{2j-1}, n_{2j}),  
# if n_{2j-1}=1, n_{2j}=0, then set f_{2j-1}=f_{2j}=0; if n_{2j-1}=0, n_{2j}=1, act SWAP gate, 
# n_{2j-1}=1, n_{2j}=0, then set f_{2j-1}=f_{2j}=0.

# 4, Even-layer evolution, if f_{2j}=1 or f_{2j+1}=1, update 01 -> 10 or 01 -> 01 is equal probability, the same for 10, 00 , 11 
#   then set f_{2j}=f_{2j+1}=1
# 5, Even-layer measurement and feedback, with probability p, measure the occupation numbers on two neighboring sites (n_{2j}, n_{2j+1}),
# if n_{2j}=0, n_{2j+1}=1, then set f_{2j}=f_{2j+1}=0; if n_{2j}=1, n_{2j+1}=0, act SWAP gate, n_{2j}=0, n_{2j+1}=1, then set f_{2j}=f_{2j+1}=0
            

using LinearAlgebra
using JLD2

f=open("read.in","r")
    l=parse(Int64,readline(f))    # # of sites even
    r=parse(Float64,readline(f)) # feedback rate
    p=parse(Float64,readline(f)) # measurement probability
close(f)

Nstep=8000
#### initial condition
bitstrings=rand(0:1,l)
fvec=ones(Int64,l)
rho_active=zeros(Float64,Nstep+1)
imbalance=zeros(Float64,Nstep+1)

function Imbalance_cal(l::Int64,bitstrings)
    imbalance=0.0
    for i=1:l
        imbalance=imbalance+abs(bitstrings[i]-bitstrings[mod(i,l)+1])
    end
    imbalance=imbalance/l
    return imbalance
end

rho_active[1]=sum(fvec)/l
imbalance[1]=Imbalance_cal(l,bitstrings)

for step=1:Nstep
    global bitstrings
    global fvec

    #### odd-bond unitary evolution
    for i=1:2:l-1 #bond i
        if fvec[i]==1 || fvec[i+1]==1 ## one of sites is active
            # if bitstrings[i] != bitstrings[i+1]  #(n_{i},n_{i+1})=(0,1) or (1,0)
            #     if rand() < 0.5 #50% probablity 01 -> 10; 10 -> 01
            #         bitstrings[i], bitstrings[i+1] = bitstrings[i+1], bitstrings[i]
            #     end
            # else #(n_{i},n_{i+1})=(0,0) or (1,1)
            #     if rand() <0.5 #50% probablity 00 -> 11, 11 -> 00
            #         bitstrings[i]=mod(bitstrings[i]+1,2) 
            #         bitstrings[i+1]=mod(bitstrings[i+1]+1,2)
            #     end
            # end
            if rand() <0.5
                bitstrings[i]=mod(bitstrings[i]+1,2) 
                bitstrings[i+1]=mod(bitstrings[i+1]+1,2)
            end
            fvec[i]=1
            fvec[i+1]=1
        end
    end

    ####  odd-bond measurement (probability p)
    #### measurement result (1,1)&(0,0), do nothing
    #### measurement result (1,0), then set f_{i}=f_{i+1}=0
    #### measurement result (0,1), probability r to act SWAP gate on site i&i+1,then set f_{i}=f_{i+1}=0
    for i=1:2:l-1
        if rand() < p
            if bitstrings[i]==1 && bitstrings[i+1]==0
                fvec[i]=0
                fvec[i+1]=0
            elseif bitstrings[i]==0 && bitstrings[i+1]==1
                if rand() < r  #probability r to act SWAP gate on site i&i+1 (0,1) -> (1,0)
                    bitstrings[i]=1
                    bitstrings[i+1]=0
                end
                fvec[i]=0
                fvec[i+1]=0
            end
        end
    end

    #### even-bond unitary evolution
    for i=2:2:l #bond i
        if fvec[i]==1 || fvec[mod(i,l)+1]==1
            # if bitstrings[i] != bitstrings[mod(i,l)+1]  #(n_{i},n_{i+1})=(0,1) or (1,0)
            #     if rand() < 0.5 #50% probability 01->10,10->01
            #         bitstrings[i], bitstrings[mod(i,l)+1] = bitstrings[mod(i,l)+1], bitstrings[i]
            #     end
            # else #(n_{i},n_{i+1})=(0,0) or (1,1)
            #     if rand() < 0.5 #50% probability 00->11, 11->00
            #         bitstrings[i]=mod(bitstrings[i]+1,2) 
            #         bitstrings[mod(i,l)+1]=mod(bitstrings[mod(i,l)+1]+1,2)
            #     end
            # end
            if rand() < 0.5
                bitstrings[i]=mod(bitstrings[i]+1,2) 
                bitstrings[mod(i,l)+1]=mod(bitstrings[mod(i,l)+1]+1,2)
            end
            fvec[i]=1
            fvec[mod(i,l)+1]=1
        end
    end

    ####  even-bond measurement (probability p)
    #### measurement result (1,1)&(0,0), do nothing
    #### measurement result (0,1), then set f_{i}=f_{i+1}=0
    #### measurement result (1,0), probability r to act SWAP gate on site i&i+1,then set f_{i}=f_{i+1}=0
    for i=2:2:l
        if rand() < p #probability p for acting on measurement n_{i} & n_{i+1}
            if bitstrings[i]==0 && bitstrings[mod(i,l)+1]==1
                fvec[i]=0
                fvec[mod(i,l)+1]=0
            elseif bitstrings[i]==1 && bitstrings[mod(i,l)+1]==0
                if rand() < r  #probability r to act SWAP gate on site i&i+1 (1,0) -> (0,1)
                    bitstrings[i]=0
                    bitstrings[mod(i,l)+1]=1
                end
                fvec[i]=0
                fvec[mod(i,l)+1]=0
            end
        end
    end

     #### observables
    rho_active[step+1]=sum(fvec)/l
    imbalance[step+1]=Imbalance_cal(l,bitstrings)

end

@save "rho_active.jld2" rho_active
@save "imbalance.jld2" imbalance
