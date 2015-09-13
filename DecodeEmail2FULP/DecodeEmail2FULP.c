/* 
* @Author: grantmcgovern
* @Date:   2015-09-11 12:19:51
* @Last Modified by:   grantmcgovern
* @Last Modified time: 2015-09-13 15:42:09
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct File_Packet {
	char * file_data;
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
	int i;
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
		printf("Invalid Number of Arguments");
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

		char *file_data = malloc(file_size + 1);
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
___global___ void caesar_cipher(char *file_data) {
	int i = 0; 
	while(file_data[i] != '\0') {
		int to_int = (int)file_data[i];
		char decrypted = (char)(to_int - 1);
	    printf("%c", decrypted);
		i++;
	}
}

/*
* MAIN
*/
int main(int argc, char *argv[]) {
	// First check command line args are valid
	check_command_line_args(argc);
	// Get file contents
	struct File_Packet packet = read_encrypted_file(argv, argc);
	
	// Decrypt
	int file_size = packet.file_size;
	int size = file_size * sizeof(char*);
	
	char *file_data = packet.file_data;

	// Local memory
	char *decrypted_file_data;
	
	// Device memory
	char *dev_decrypted_file_data;
	char *dev_file_data;
	
	// Allocate memory on the GPU
	cudaMalloc((void**)&dev_file_data, size);
	cudaMalloc((void**)&dev_decrypted_file_data, size);

	cudaMemcpy(dev_file_data, file_data, size, cudaMemcpyHostToDevice);

	caesar_cipher<<<1, file_size>>>(dev_file_data, dev_decrypted_file_data);

	cudaThreadSynchronize();

	cudaMemcpy(decrypted_file_data, dev_decrypted_file_data, size, cudaMemcpyDeviceToHost);
	
	// Deallocate memory 
	cudaFree(dev_decrypted_file_data);

	exit(0);
	//caesar_cipher(packet.file_data, packet.file_size);
    //return 0;
}
