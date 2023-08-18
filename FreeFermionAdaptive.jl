#### Free fermions under adaptive quantum dynamics####
#### periodic boundary condition
using LinearAlgebra
using JLD2

f=open("read.in","r")
    l=parse(Int64,readline(f))    # # of sites even
    r=parse(Float64,readline(f)) # feedback rate
    p=parse(Float64,readline(f)) # measurement probability
close(f)


sublength=trunc(Int,l/5)
Nstep=300
alphamat=zeros(ComplexF64,2*l,l) #corresponding to annihilation operator
#### initial observables
rho_active=zeros(Float64,Nstep+1)
imbalance=zeros(Float64,Nstep+1)
renyi=zeros(Float64,Nstep+1)

#### initialization: t=0 product state; all sites active f_i=1 ∀ i
#### choose |ψ(t=0)⟩=|000000...000⟩, corresponding α matrix 
for i=1:l
    alphamat[i,i]=1.0
end
fvec=ones(Int64,l) # 1:active, 0:inactive; initial all sites active


function U1gate(i::Int64,l::Int64) #U1gate act on site i and i+1
    U1gate=zeros(ComplexF64,2*l,2*l)
    U1gate[i,mod(i,l)+1]=1.0
    U1gate[mod(i,l)+1,i]=1.0
    U1gate[i+l,mod(i,l)+1+l]=-1.0
    U1gate[mod(i,l)+1+l,i+l]=-1.0
    return U1gate
end

function U2gate(i::Int64,l::Int64) #U2gate act on site i and i+1
    U2gate=zeros(ComplexF64,2*l,2*l)
    U2gate[i,mod(i,l)+1+l]=1.0
    U2gate[mod(i,l)+1,i+l]=-1.0
    U2gate[i+l,mod(i,l)+1]=-1.0
    U2gate[mod(i,l)+1+l,i]=1.0
    return U2gate
end

function Proboccupy(i::Int64,l::Int64,alphamat) # probability of measurement's result n_i=1
    #### ⟨ψ|n_i|ψ⟩=C_{i+L,i+L}=∑_{k} |(α_{k})_{i+L}|^2
    probone=0.0
    for k=1:l
        probone=probone+(abs(alphamat[i+l,k]))^2
    end
    return probone
end

function Oneupdate(i::Int64,l::Int64,alphamat)
    vectemp=abs.(alphamat[i+l,:])
    i0=findmax(vectemp)[2]
    for j=1:l
        if j == i0
            continue
        else
            alphamat[:,j]=alphamat[:,j]-(alphamat[i+l,j]/alphamat[i+l,i0])*alphamat[:,i0]
            alphamat[i,j]=0.0
        end
    end
    alphamat[:,i0]=zeros(Float64,2*l)
    alphamat[i+l,i0]=1.0
    #### perform  Gram-Schmidt method to orthgonalize alphamat
    #### directly use QR decomposition
    F=qr(alphamat)
    alphamat=F.Q*Matrix(1.0I,l,l)
    return alphamat
end

function Zeroupdate(i::Int64,l::Int64,alphamat)
    vectemp=abs.(alphamat[i,:])
    i0=findmax(vectemp)[2]
    for j=1:l
        if j == i0
            continue
        else
            alphamat[:,j]=alphamat[:,j]-(alphamat[i,j]/alphamat[i,i0])*alphamat[:,i0]
            alphamat[i+l,j]=0.0
        end
    end
    alphamat[:,i0]=zeros(Float64,2*l)
    alphamat[i,i0]=1.0
    #### perform  Gram-Schmidt method to orthgonalize alphamat
    #### directly use QR decomposition
    F=qr(alphamat)
    alphamat=F.Q*Matrix(1.0I,l,l)
    return alphamat
end

function SWAPodd(i::Int64,l::Int64,alphamat)  
    #### perform SWAP gate on site i&i+1 i odd (0,1) -> (1,0)
    #### exchange row i & i+1 ; i+l &i+1+l
    alphamat[i,:],alphamat[i+1,:],alphamat[i+l,:],alphamat[i+1+l,:]=alphamat[i+1,:],alphamat[i,:],alphamat[i+1+l,:],alphamat[i+l,:]
    #### orthonormalizing
    # F=qr(alphamat)
    # alphamat=F.Q*Matrix(1.0I,l,l)
    return alphamat
end

function SWAPeven(i::Int64,l::Int64,alphamat)  
    #### perform SWAP gate on site i&i+1  i even (1,0) -> (0,1)
    #### PBC
    #### exchange row i & i+1 ; i+l &i+1+l
    alphamat[i,:],alphamat[mod(i,l)+1,:],alphamat[i+l,:],alphamat[mod(i,l)+1+l,:]=alphamat[mod(i,l)+1,:],alphamat[i,:],alphamat[mod(i,l)+1+l,:],alphamat[i+l,:]
    #### orthonormalizing
    # F=qr(alphamat)
    # alphamat=F.Q*Matrix(1.0I,l,l)
    return alphamat
end

function Imbalance_cal(l::Int64,corr)
    imbalance=0.0
    for i=1:l
        imbalance=imbalance+abs(corr[i+l,i+l]-corr[mod(i,l)+1+l,mod(i,l)+1+l])
    end
    imbalance=imbalance/l
    return imbalance
end

function Renyi_cal(l::Int64,sublength::Int64,corr)
    corr_sub=zeros(ComplexF64,2*sublength,2*sublength)
    corr_sub[1:sublength,1:sublength]=corr[1:sublength,1:sublength]
    corr_sub[1:sublength,sublength+1:2*sublength]=corr[1:sublength,l+1:l+sublength]
    corr_sub[sublength+1:2*sublength,1:sublength]=corr[1+l:sublength+l,1:sublength]
    corr_sub[sublength+1:2*sublength,sublength+1:2*sublength]=corr[1+l:sublength+l,1+l:sublength+l]
    renyi_temp=-0.5*tr(log(corr_sub*corr_sub+(Matrix(1.0I,2*sublength,2*sublength)-corr_sub)*(Matrix(1.0I,2*sublength,2*sublength)-corr_sub)))
    return real(renyi_temp)
end

rho_active[1]=sum(fvec)/l
corr=alphamat*adjoint(alphamat)
imbalance[1]=Imbalance_cal(l,corr)
renyi[1]=Renyi_cal(l,sublength,corr)

for step=1:Nstep   # # of steps of evolution until steady state
    global alphamat
    global fvec
    local corr

    #### odd-bond unitary evolution
    for i=1:2:l-1 #bond i
        if fvec[i]==1 || fvec[i+1]==1
            #### equal probability to act U_1 or U_2
            if rand() < 0.5 # act U_1 on sites i and i+1
                alphamat=exp(-1im*pi/4*U1gate(i,l))*alphamat
            else
                alphamat=exp(-1im*pi/4*U2gate(i,l))*alphamat
            end
            fvec[i]=1
            fvec[i+1]=1
        end
    end

    ####  odd-bond measurement (probability p)
    for i=1:2:l-1
        if rand() < p #probability p for acting on measurement n_{i} & n_{i+1}
            temp=[0,0]
            probone1=Proboccupy(i,l,alphamat) # prob of n_{i}=1
            if rand() < probone1
                alphamat=Oneupdate(i,l,alphamat)
                temp[1]=1
            else
                alphamat=Zeroupdate(i,l,alphamat)
            end
            probone2=Proboccupy(i+1,l,alphamat) # prob of n_{i+1}=1
            if rand() < probone2
                alphamat=Oneupdate(i+1,l,alphamat)
                temp[2]=1
            else
                alphamat=Zeroupdate(i+1,l,alphamat)
            end
            #### additional operations determinated by measurement results
            #### measurement result (1,1)&(0,0), do nothing
            #### measurement result (1,0), then set f_{i}=f_{i+1}=0
            #### measurement result (0,1), probability r to act SWAP gate on site i&i+1,then set f_{i}=f_{i+1}=0
            if temp == [1,0]
                fvec[i]=0
                fvec[i+1]=0
            elseif temp == [0,1]
                if rand() < r  #probability r to act SWAP gate on site i&i+1 (0,1) -> (1,0)
                    alphamat=SWAPodd(i,l,alphamat)
                end
                fvec[i]=0
                fvec[i+1]=0
            end
        end
    end

    #### even-bond unitary evolution
    for i=2:2:l #bond i
        if fvec[i]==1 || fvec[mod(i,l)+1]==1
            #### equal probability to act U_1 or U_2
            if rand() < 0.5 # act U_1 on sites i and i+1
                alphamat=exp(-1im*pi/4*U1gate(i,l))*alphamat
            else
                alphamat=exp(-1im*pi/4*U2gate(i,l))*alphamat
            end
            fvec[i]=1
            fvec[mod(i,l)+1]=1
        end
    end

    ####  even-bond measurement (probability p)
    for i=2:2:l
        if rand() < p #probability p for acting on measurement n_{i} & n_{i+1}
            temp=[0,0]
            probone1=Proboccupy(i,l,alphamat) # prob of n_{i}=1
            if rand() < probone1
                alphamat=Oneupdate(i,l,alphamat)
                temp[1]=1
            else
                alphamat=Zeroupdate(i,l,alphamat)
            end
            probone2=Proboccupy(mod(i,l)+1,l,alphamat) # prob of n_{i+1}=1
            if rand() < probone2
                alphamat=Oneupdate(mod(i,l)+1,l,alphamat)
                temp[2]=1
            else
                alphamat=Zeroupdate(mod(i,l)+1,l,alphamat)
            end
            #### measurement result (1,1)&(0,0), do nothing
            #### measurement result (0,1), then set f_{i}=f_{i+1}=0
            #### measurement result (1,0), probability r to act SWAP gate on site i&i+1,then set f_{i}=f_{i+1}=0
            if temp == [0,1]
                fvec[i]=0
                fvec[mod(i,l)+1]=0
            elseif temp == [1,0]
                if rand() < r  #probability r to act SWAP gate on site i&i+1 (1,0) -> (0,1)
                    alphamat=SWAPeven(i,l,alphamat)
                end
                fvec[i]=0
                fvec[mod(i,l)+1]=0
            end

        end
    end

    #### observables
    rho_active[step+1]=sum(fvec)/l
    corr=alphamat*adjoint(alphamat)
    imbalance[step+1]=Imbalance_cal(l,corr)
    renyi[step+1]=Renyi_cal(l,sublength,corr)

end

@save "rho_active.jld2" rho_active
@save "imbalance.jld2" imbalance
@save "renyi.jld2" renyi

