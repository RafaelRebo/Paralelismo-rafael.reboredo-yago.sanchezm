#include <stdio.h>
#include <math.h>
#include <string.h>
#include "/usr/include/x86_64-linux-gnu/mpi/mpi.h" //Ver la ruta de mpi.h

#define TAG 2004

int MPI_Flattree(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm){
    int numprocs, rank;
    double partial, total;
    memcpy(&total, sendbuf, count * sizeof(double));

    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (comm == MPI_COMM_NULL) return MPI_ERR_COMM;
    if (count < 0) return MPI_ERR_COUNT;
    if (sendbuf == NULL) return MPI_ERR_BUFFER;
    if (op != MPI_SUM) return MPI_ERR_OP; //Solo soporta suma
    if (rank < 0 || rank > numprocs-1) return MPI_ERR_RANK;
    if (datatype != MPI_DOUBLE) return MPI_ERR_TYPE;

    if(rank==root){
        for(int i=1;i<numprocs;i++){
            MPI_Recv(&partial,count,datatype,MPI_ANY_SOURCE,TAG,comm,MPI_STATUS_IGNORE);
            total+=partial;
        }
        memcpy(recvbuf,&total,sizeof(double));
    }
    else{
        MPI_Send(sendbuf,count,datatype,root,TAG,comm);
    }
    return MPI_SUCCESS;
}

int MPI_BinomialBCast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    int rank,numprocs,it=2;

    MPI_Comm_size(comm,&numprocs);
    MPI_Comm_rank(comm,&rank);

    if (comm == MPI_COMM_NULL) return MPI_ERR_COMM;
    if (count < 0) return MPI_ERR_COUNT;
    if (rank < 0 || rank > numprocs-1) return MPI_ERR_RANK;

    if (rank != root) MPI_Recv(buffer,count,datatype,MPI_ANY_SOURCE,TAG,comm,MPI_STATUS_IGNORE);
    else MPI_Send(buffer,count,datatype,1,TAG,comm);

    for(int i=2;;i++){
        if(rank< it){
            if (it+rank >= numprocs) break;
            printf("Se envia al proceso -> %i | el dato -> %i | FROM: %i\n", 2*(i-1)+rank,*(int *)buffer,rank);
            MPI_Send(buffer,count,datatype,2*(i-1)+rank ,TAG,comm);
        }
        it=it*2;
    }
    return 0;
}

int main(int argc, char *argv[])
{
    int done = 0, n, output;
    int numprocs, rank;
    double total_sum, pi;
    double PI25DT = 3.141592653589793238462643;

    MPI_Init(&argc,&argv);

    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    while (!done){
        if(rank == 0){
            printf("Enter the number of intervals: (0 quits) \n");
            scanf("%d",&n);
        }
        //MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_BinomialBCast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
        printf("%d - %d\n", rank, n);

        if(n==0) break;

        double partial_sum, h, x;
        h   = 1.0 / (double) n;
        partial_sum = 0.0;
        for (int i = rank+1; i <= n; i+=numprocs) {
            x = h * ((double)i - 0.5);
            partial_sum += 4.0 / (1.0 + x*x);
        }

        //MPI_Reduce(&partial_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if((output=MPI_Flattree(&partial_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD))!=0){
            printf("Process exited with error %d",output);
            return -1;
        }
        if(rank==0){
            pi=h*total_sum;
            printf("pi is approximately %.16f, Error is %.16f\n", pi, fabs(pi - PI25DT));
        }
    }

    MPI_Finalize();
    return 0;
}
