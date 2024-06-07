import torch


def compute_average_triplet_loss(
    model,
    anchor_encode,
    pos_list,
    neg_list,
    loss,
    device
):
    """
    Compute the average triplet loss for a given anchor encoding, positive and negative samples.
    """
    batch_size = anchor_encode.shape[0]
    num_samples = pos_list.shape[1]

    pos_list = pos_list.reshape((-1, *pos_list.shape[2:])).to(device)
    neg_list = neg_list.reshape((-1, *neg_list.shape[2:])).to(device)

    pos_encodes = model.encoder(pos_list).split(num_samples, dim=0)  # model.encoder is tuple (mean, var)
    neg_encodes = model.encoder(neg_list).split(num_samples, dim=0)

    pos_encodes = torch.stack(pos_encodes).permute(1, 0, 2)
    neg_encodes = torch.stack(neg_encodes).permute(1, 0, 2)

    anchor_encode = torch.stack([anchor_encode] * num_samples)

    return loss(anchor_encode, pos_encodes, neg_encodes)


def compute_reconstruction_loss(anchor, anchor_decode, loss):
    """
    Compute the reconstruction loss for the decoded anchor tensor.
    """
    return loss(anchor, anchor_decode)


def compute_loss(
    model,
    anchor,
    anchor_encode,
    anchor_decode,
    pos_transcriptomics_list,
    neg_transcriptomics_list,
    pos_morphology_list,
    neg_morphology_list,
    transcriptomics_weight,
    morphology_weight,
    triplet_loss,
    triplet_alpha,
    rec_loss,
    rec_alpha,
    obsm_transcriptomics_dim,
    device
):
    """
    Compute the total loss given a model, anchor encoding and decoding, and various inputs and coefficients.
    """
    triplet_loss_transcriptomics = compute_average_triplet_loss(
        model,
        anchor_encode,
        pos_transcriptomics_list,
        neg_transcriptomics_list,
        triplet_loss,
        device) if transcriptomics_weight > 0 and triplet_alpha else 0

    triplet_loss_morphology = compute_average_triplet_loss(
        model,
        anchor_encode,
        pos_morphology_list,
        neg_morphology_list,
        triplet_loss,
        device) if morphology_weight > 0 and triplet_alpha else 0

    rec_loss_transcriptomics = compute_reconstruction_loss(
        anchor[:, :obsm_transcriptomics_dim],
        anchor_decode[:, :obsm_transcriptomics_dim],
        rec_loss) if transcriptomics_weight > 0 and rec_alpha else 0

    rec_loss_morphology = compute_reconstruction_loss(
        anchor[:, obsm_transcriptomics_dim:],
        anchor_decode[:, obsm_transcriptomics_dim:],
        rec_loss) if morphology_weight > 0 and rec_alpha else 0

    total_loss = transcriptomics_weight * (rec_alpha * rec_loss_transcriptomics + triplet_alpha * triplet_loss_transcriptomics) + \
        morphology_weight * (rec_alpha * rec_loss_morphology + triplet_alpha * triplet_loss_morphology)

    return total_loss, rec_loss_transcriptomics, rec_loss_morphology, triplet_loss_transcriptomics, triplet_loss_morphology
