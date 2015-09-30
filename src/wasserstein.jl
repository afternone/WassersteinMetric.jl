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

#get wasserstein distance matrix
function _getwsdm(edge,ghh=1,wei=ones(size(edge,1)))
	wei=wei[edge[:,1].<edge[:,2]]
    edge=[edge[edge[:,1].<edge[:,2],1] edge[edge[:,1].<edge[:,2],2]]
    N=maximum(edge)
    adj=sparse([edge[:,1],N],[edge[:,2],N],[wei,0])
    adj=adj+adj'
    adjh=adj^ghh
    dist=sparse([edge[:,1],N],[edge[:,2],N],[ones(size(edge,1)),0])
    dist=dist+dist'
	sp=floyd_warshall(1./dist)
    wsm=spzeros(N,N)
    p=Progress(N*N, 1, "Computing initial pass...")
    for jw=1:N,iw=jw+1:N
		if adj[iw,jw]>0
			xneis=find(adjh[iw,:])
			xsize=size(xneis,1)
			xwei=findnz(adjh[iw,:])[3]./sum(findnz(adjh[iw,:])[3])
			yneis=find(adjh[jw,:])
			ysize=size(yneis,1)
			ywei=findnz(adjh[jw,:])[3]./sum(findnz(adjh[jw,:])[3])
			f=[xwei;ywei]
			b=zeros(xsize,ysize)
			for j=1:ysize,i=1:xsize
				 b[i,j]=sp[xneis[i],yneis[j]]
			end
			b=reshape(b',xsize*ysize,1)
			A=zeros(0,xsize+ysize)
			IY=eye(ysize)
			for er=1:xsize
				U=zeros(ysize,xsize)
				U[:,er]=1
				A=vcat(A,hcat(U,IY))
			end
			wsm[iw,jw]=linprog(f,A,b)
		end
	next!(p)
    end
    sum(wsm)/size(edge,1)
end

function getwsdm(edge, ghh=1)
    N=max(maximum(findn(edge)[1]), maximum(findn(edge)[2]))
    adj=edge+edge'
    adjh=adj^ghh
    dist=sparse(findn(edge)[1],findn(edge)[2],ones(nnz(edge)), N, N)
    dist=dist+dist'
	sp=floyd_warshall(1./dist)
    wsm=spzeros(N,N)
    p=Progress(N*N, 1, "Computing initial pass...")
    for jw=1:N,iw=jw+1:N
		if adj[iw,jw]>0
			xneis=find(adjh[iw,:])
			xsize=size(xneis,1)
			xwei=findnz(adjh[iw,:])[3]./sum(findnz(adjh[iw,:])[3])
			yneis=find(adjh[jw,:])
			ysize=size(yneis,1)
			ywei=findnz(adjh[jw,:])[3]./sum(findnz(adjh[jw,:])[3])
			f=[xwei;ywei]
			b=zeros(xsize,ysize)
			for j=1:ysize,i=1:xsize
				 b[i,j]=sp[xneis[i],yneis[j]]
			end
			b=reshape(b',xsize*ysize,1)
			A=zeros(0,xsize+ysize)
			IY=eye(ysize)
			for er=1:xsize
				U=zeros(ysize,xsize)
				U[:,er]=1
				A=vcat(A,hcat(U,IY))
			end
			wsm[iw,jw]=linprog(f,A,b)
		end
	next!(p)
    end
    sum(wsm)/nnz(edge)
end
