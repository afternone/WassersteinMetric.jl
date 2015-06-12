function linprog(f,A,b)
m=Model()
@defVar(m,x[1:size(A,2)]>=0)
@setObjective(m,Min,dot(f,x))
for j=1:size(A,1)
        @addConstraint(m,sum{A[j,i]*x[i],i=1:size(A,2)}>=b[j])
end
status=solve(m)
getObjectiveValue(m)
end

function get_wasserstein_metric(cityname)
    edgelist=readdlm(cityname,Int64)
    N=maximum(edgelist)
    adjm=sparse(edgelist[:,1],edgelist[:,2],1)
    distm=1./full(adjm)
    spm=floyd_warshall(distm)
    wasserm=zeros(N,N)
    p = Progress(N*N, 1, "Computing initial pass...")
    for iw=1:N
        for jw=iw+1:N
            xneis=findn(adjm[iw,:])[2]
            yneis=findn(adjm[jw,:])[2]
            xsize=length(xneis)
            ysize=length(yneis)
            minsize=min(xsize,ysize)
            if spm[iw,jw]==1&&minsize>1
                wasserm[iw,jw]=1.0/(minsize-1)
                wasserm[jw,iw]=wasserm[iw,jw]
            else
                f=Array(Float64,xsize+ysize)
                f[1:xsize]=1.0/xsize
                f[xsize+1:xsize+ysize]=1.0/ysize
                b=Array(Float64,xsize*ysize)
                step=0
                for i=1:xsize
                    for j=1:ysize
                        b[step=step+1]=spm[xneis[i],yneis[j]]
                    end
                end
                A=zeros(0,xsize+ysize)
                IY=eye(ysize,ysize)
                for i=1:xsize
                    U=zeros(ysize,xsize)
                    U[:,i]=1
                    A=vcat(A,hcat(U,IY))
                end
                wasserm[iw,jw]=linprog(f,A,b)
                wasserm[jw,iw]=wasserm[iw,jw]
            end
            next!(p)
        end
    end
    wasserm
end