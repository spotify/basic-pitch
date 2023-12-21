$(document).ready(function() {
    $('form#transcribe_form').submit(function(event) { 
        event.preventDefault();
        var formData = new FormData(this);
        
        transcribe_audio(formData);
    });
});

function transcribe_audio(formData) {
    // disable transcribe button and show loading animation
    $('#transcribe_button').prop('disabled', true);
    $("#input_select_model").prop('disabled', true);
    $("#model_results").hide();
    $("#results_error").hide();
    $("#error_message").html('');
    $('#results_loading').show();

    // call python script to transcribe audio
    server_request('http://127.0.0.1:8000/transcribe', 'POST', formData).then(function(response) {
        switch (response.status) {
            case 'success':
                get_midi_file(response.data);
                break;
            case 'failure':
                display_error(response.data);
                break;
            case 'error':
                display_error(response.data);
                break;
            default:
                console.log(response)
                break;
        }
    });
}

function get_midi_file(file_name) { 
    // full file path
    var midi_file_url = "http://127.0.0.1:8000/static/" + file_name;


    display_midi(midi_file_url);
}

function display_midi(data) {
    $('#transcribe_button').prop('disabled', false);
    $("#input_select_model").prop('disabled', false);
    $("#results_error").hide();
    $("#error_message").html('');
    $('#results_loading').hide();
    
    load_midi(data)
    $("#model_results").show();
}

function display_error(message) {
    $('#transcribe_button').prop('disabled', false);
    $("#input_select_model").prop('disabled', false);
    $("#error_message").html(message);
    $("#results_error").show();
    $('#results_loading').hide();
}

function server_request(url, method, data = null) {
    return new Promise(function(resolve) {
        $.ajax({
            type: method,
            url: url,
            data: data,
            processData: false, // Prevent serialization of the FormData object
            contentType: false, // Let the browser set the correct content type for FormData
            success: function (response, status) {
                // AJAX Success, server responded
                console.log(response)
                // Response successfully parsed as JSON, PHP compiled and caught all errors
                resolve(response);
            },
            error: function (XMLHttpRequest, textStatus, errorThrown) {
                // AJAX Error, server did not respond
                response = {
                    status: 'error', 
                    data: "AJAX ERROR (" + textStatus + ")\n" + errorThrown
                };
                resolve(response);
            },
            statusCode: {
                500: function() {
                    alert("Internal server error. Please try again later.");
                 }
              },
        })
    });
}