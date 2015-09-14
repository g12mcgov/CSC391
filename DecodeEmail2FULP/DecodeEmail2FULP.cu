/* 
* @Author: grantmcgovern
* @Date:   2015-09-11 12:19:51
* @Last Modified by:   grantmcgovern
* @Last Modified time: 2015-09-13 15:42:09
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
* File_Packet 
*
* Contains a small data packet of
* the file info (data + size) to 
* help with dynamic allocation.
*
*/
struct File_Packet {
	char *file_data;
	int file_size;
};

/*
* get_filename_length(char *[])
* 
* Computes the length of command line
* argument filename to store the filename
* as a string, and null-terminate it.
*
*/
int get_filename_length(char *filename[]) {
	int i = 0;
	while(filename[1][i] != '\0')
		i++;
	return i;
}

/*
* check_command_line_args(int)
*
* Checks to see whether used used proper
* number of command line arguments.
*/
void check_command_line_args(int argc) {
	// Ensure command line args are limited to only 1
	if(argc > 2 || argc == 1) {
		printf("Invalid Number of Arguments\n");
		exit(1);
	}
}

/*
* read_encrypted_file(char*, int)
*
* Takes command line args passed from main
* and opens the file, reading the data, then
* bulding a character array.
*/
struct File_Packet read_encrypted_file(char *args[], int length) {
	int filename_length = get_filename_length(args);
	// printf("%d\n", filename_length);
	char filename[filename_length + 1];
	// Null terminate the end to ensure no weird chars
	filename[filename_length] = '\0';
	// Prevents buffer overflow, copies filename
	strncpy(filename, args[1], filename_length);

	/* 
	* Read in file content, use fseek()
	* to get file size, and dynamically
	* allocate a string.
	*/
	FILE *file = fopen(filename, "rb");

	// Check if file exits
	if(file) {
		fseek(file, 0, SEEK_END);
		long file_size = ftell(file);
		fseek(file, 0, SEEK_SET);

		char *file_data = (char *)(malloc(file_size + 1));
		fread(file_data, file_size, 1, file);
		fclose(file);

		file_data[file_size] = 0;

		struct File_Packet packet;
		packet.file_data = file_data;
		packet.file_size = file_size;

		return packet;
	}
	else {
		printf("%s\n", "File does not exist");
		exit(1);
	}
}

/*
* caesar_cipher(char*)
*
* Takes a character array of the file contents
* and converts each character to its decrypted
* state by first casting to int, decrementing by
* 1, then casting back to a char.
*/
__global__ void caesar_cipher(char *file_data, char *dev_decrypted_file_data) {
	// char decrypted_text[file_size];
	int i = threadIdx.x;
	//while(file_data[i] != '\0') {
	int to_int = (int)file_data[i];
	char decrypted = (char)(to_int - 1);
	dev_decrypted_file_data[i] = decrypted;
		//i++;
	//}
	// Null terminate it for check
	//dev_decrypted_file_data[i + 1] = '\0';
}

/*
* print_decrypted_message(char *)
*
* Recieves the memory block back from CUDA,
* and prints the decrypted message.
*/
void print_decrypted_message(char *decrypted_file_data) {
	printf("%s\n", decrypted_file_data);
	exit(0);
}

/*
* MAIN
*/
int main(int argc, char *argv[]) {
	// First check command line args are valid
	check_command_line_args(argc);
	// Get file contents
	struct File_Packet packet = read_encrypted_file(argv, argc);
	
	// Get file length (chars)
	int file_size = packet.file_size;
	// Compute size of memory block we'll need
	int size = file_size * sizeof(char);
	
	// Local memory
	char *file_data = packet.file_data;
	char decrypted_file_data[file_size];
	
	// Device memory
	char *dev_file_data;
	char *dev_decrypted_file_data;
	
	// Allocate memory on the GPU
	cudaMalloc((void**)&dev_file_data, size);
	cudaMalloc((void**)&dev_decrypted_file_data, size);

	cudaMemcpy(dev_file_data, file_data, size, cudaMemcpyHostToDevice);

	// Decrypt the message on the GPU
	caesar_cipher<<<1, file_size>>>(dev_file_data, dev_decrypted_file_data);

	// Not sure if we need this, since we're only running on 1 thread
	cudaThreadSynchronize();

	cudaMemcpy(decrypted_file_data, dev_decrypted_file_data, size, cudaMemcpyDeviceToHost);
	
	// Check we've decrypted
	print_decrypted_message(decrypted_file_data);

	// Deallocate memory on CUDA
	cudaFree(dev_file_data);
	cudaFree(dev_decrypted_file_data);

	// Exit with success
	exit(0);
}
